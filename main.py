import os
import random
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Iterable, List, Tuple
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from telegram import (
    Update,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    JobQueue,
    CommandHandler,
    ConversationHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    Defaults
)

load_dotenv()

TZ = ZoneInfo("Europe/Moscow")
DB_PATH = Path("bot.db")
CONTENT_PATH = Path("content.txt")

# =========================
# Константы конфигурации
# =========================

MORNING_TIME = time(8, 30)
DAY_TIME = time(14, 0)
EVENING_TIME = time(21, 0)
WEEKLY_REPORT_HOUR = 19
AWARENESS_HOUR = 20
EXCLUDE_DAYS = 14
ADVICE_PROB_BASE = 0.6
ADAPTIVE_PROB = True


def now_msk() -> datetime:
    return datetime.now(TZ)


# =========================
# DB helpers
# =========================

def get_connection(db_path: Path | str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Создать таблицы, если их нет."""
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          chat_id INTEGER UNIQUE NOT NULL,
          start_date TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS advices (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          kind TEXT NOT NULL,
          time_window TEXT NOT NULL,
          text TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS advice_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          advice_id INTEGER,
          kind TEXT NOT NULL,
          time_window TEXT NOT NULL,
          sent_at TEXT NOT NULL,
          rating INTEGER,
          not_relevant INTEGER DEFAULT 0,
          planned INTEGER DEFAULT 0,
          helped INTEGER DEFAULT 0,
          rest_type TEXT,
          rest_category TEXT,
          rest_note TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS evening_checkins (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          sent_at TEXT NOT NULL,
          mood INTEGER NOT NULL,
          workload INTEGER NOT NULL,
          productivity INTEGER NOT NULL,
          note TEXT
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS awareness (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          sent_at TEXT NOT NULL,
          response TEXT NOT NULL
        );
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS planned_breaks (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id INTEGER NOT NULL,
          remind_at TEXT NOT NULL,
          created_at TEXT NOT NULL
        );
        """
    )

    conn.commit()


def get_or_create_user(conn: sqlite3.Connection, chat_id: int) -> int:
    """
    Вернуть user.id по chat_id, при отсутствии — создать.
    start_date = сейчас по MSK (ISO с зоной).
    """
    cur = conn.cursor()
    cur.execute("SELECT id FROM users WHERE chat_id = ?", (chat_id,))
    row = cur.fetchone()
    if row:
        return int(row["id"])

    now = now_msk().isoformat()
    cur.execute(
        "INSERT INTO users (chat_id, start_date) VALUES (?, ?)",
        (chat_id, now),
    )
    conn.commit()
    return int(cur.lastrowid)


def all_users(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Вернуть всех пользователей."""
    cur = conn.cursor()
    cur.execute("SELECT * FROM users ORDER BY id")
    return list(cur.fetchall())


# =========================
# Content parsing
# =========================

@dataclass
class AdviceRecord:
    kind: str  # "advice" | "support"
    time_window: str  # "morning" | "day" | "evening" | "any"
    text: str


def parse_content_text(content: str) -> List[AdviceRecord]:
    """
    Парсер content.txt.
    """
    records: List[AdviceRecord] = []

    current_kind: str | None = None
    current_time_window: str | None = None

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith("[") and line.endswith("]"):
            inner = line[1:-1].strip()
            parts = [p.strip() for p in inner.split("/")]

            if len(parts) < 2:
                current_kind = None
                current_time_window = None
                continue

            kind, time_window = parts[0], parts[1]

            if kind not in ("advice", "support"):
                current_kind = None
                current_time_window = None
                continue

            if time_window not in ("morning", "day", "evening", "any"):
                current_kind = None
                current_time_window = None
                continue

            current_kind = kind
            current_time_window = time_window
            continue

        if line.startswith(("-", "•")) and current_kind and current_time_window:
            text = line[1:].lstrip()
            if text:
                records.append(
                    AdviceRecord(
                        kind=current_kind,
                        time_window=current_time_window,
                        text=text,
                    )
                )
            continue

    return records


def load_content_from_file(path: Path = CONTENT_PATH) -> List[AdviceRecord]:
    if not path.exists():
        return []
    content = path.read_text(encoding="utf-8")
    return parse_content_text(content)


def import_advices(
        conn: sqlite3.Connection,
        records: Iterable[AdviceRecord],
        *,
        clear_existing: bool = True,
) -> int:
    cur = conn.cursor()
    if clear_existing:
        cur.execute("DELETE FROM advices")

    to_insert = [(r.kind, r.time_window, r.text) for r in records]

    if to_insert:
        cur.executemany(
            "INSERT INTO advices (kind, time_window, text) VALUES (?, ?, ?)",
            to_insert,
        )

    conn.commit()
    return len(to_insert)


def advices_empty(conn: sqlite3.Connection) -> bool:
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) AS cnt FROM advices")
    row = cur.fetchone()
    return bool(row["cnt"] == 0)


# =========================
# Probability logic
# =========================

@dataclass
class ProbContext:
    is_newbie: bool
    help_rate: float | None
    advice_count: int
    helped_advice_count: int
    yesterday_heavy_and_low: bool
    missed_checkins_2days: bool


def compute_day_advice_prob(
        ctx: ProbContext,
        base: float = ADVICE_PROB_BASE,
        adaptive: bool = ADAPTIVE_PROB,
) -> float:
    if ctx.is_newbie:
        return 0.5

    p = base

    if adaptive:
        if ctx.help_rate is not None and ctx.help_rate >= 0.5:
            p = min(0.9, p + 0.2)

        if ctx.advice_count >= 2 and ctx.helped_advice_count == 0:
            p = max(0.3, p - 0.3)

        if ctx.yesterday_heavy_and_low:
            p = min(0.9, p + 0.1)

        if ctx.missed_checkins_2days:
            p = 0.4

    if p < 0.2:
        p = 0.2
    if p > 0.9:
        p = 0.9

    return p


def choose_day_message_kind(p: float) -> str:
    """С вероятностью p возвращает 'advice'."""
    r = random.random()
    return "advice" if r < p else "support"


# =========================
# Advice picking with EXCLUDE_DAYS
# =========================

def pick_advice_for_user(
        conn: sqlite3.Connection,
        user_id: int,
        *,
        kind: str,
        time_window: str,
        exclude_days: int = EXCLUDE_DAYS,
) -> sqlite3.Row | None:
    cur = conn.cursor()

    params = [kind, time_window]
    query = "SELECT * FROM advices WHERE kind=? AND time_window=?"

    if exclude_days > 0:
        cutoff = now_msk() - timedelta(days=exclude_days)
        cur.execute(
            """
            SELECT DISTINCT advice_id
            FROM advice_events
            WHERE user_id=? AND kind=? AND time_window=? AND sent_at>=? AND advice_id IS NOT NULL
            """,
            (user_id, kind, time_window, cutoff.isoformat()),
        )
        blocked = [int(r["advice_id"]) for r in cur.fetchall()]

        if blocked:
            placeholders = ",".join("?" for _ in blocked)
            query += f" AND id NOT IN ({placeholders})"
            params.extend(blocked)

    cur.execute(query, params)
    rows = cur.fetchall()
    if not rows:
        return None
    return random.choice(rows)


def create_advice_event(
        conn: sqlite3.Connection,
        user_id: int,
        *,
        advice_id: int | None,
        kind: str,
        time_window: str,
        sent_at: datetime | None = None,
) -> int:
    """
    Создать запись в advice_events и вернуть её id.
    kind/time_window дублируем, даже если есть advice_id, чтобы не ломать отчёты, если текст когда-то изменится.
    """
    if sent_at is None:
        sent_at = now_msk()

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO advice_events (
          user_id, advice_id, kind, time_window, sent_at
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (user_id, advice_id, kind, time_window, sent_at.isoformat()),
    )
    conn.commit()
    return int(cur.lastrowid)


def pick_support_message(
        conn: sqlite3.Connection,
        *,
        time_window: str,
) -> sqlite3.Row | None:
    """
    Выбрать случайную поддержку (kind='support') для указанного окна времени.
    Повторы поддержки не ограничиваем EXCLUDE_DAYS (по ТЗ горизонт только для советов).
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM advices WHERE kind='support' AND time_window=?",
        (time_window,),
    )
    rows = cur.fetchall()
    if not rows:
        return None
    return random.choice(rows)


def pick_support_any(conn: sqlite3.Connection) -> sqlite3.Row | None:
    """
    Фолбэк-поддержка из [support/any].
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT * FROM advices WHERE kind='support' AND time_window='any'"
    )
    rows = cur.fetchall()
    if not rows:
        return None
    return random.choice(rows)


def build_prob_context_for_user(conn: sqlite3.Connection, user_id: int) -> ProbContext:
    """
    Построить ProbContext по последним дням активности пользователя.
    Правила:
      - is_newbie: ≤ 3 суток с момента start_date;
      - help_rate/advices по последним 3 суткам;
      - yesterday_heavy_and_low: вчерашний чек-ин с workload>=4 и mood<=3;
      - missed_checkins_2days: за последние 2 суток нет ни одного чек-ина.
    """
    cur = conn.cursor()

    # is_newbie
    cur.execute("SELECT start_date FROM users WHERE id=?", (user_id,))
    row = cur.fetchone()
    if row is None:
        # На всякий случай — если пользователь исчез, считаем новичком.
        is_newbie = True
    else:
        start_dt = datetime.fromisoformat(row["start_date"]).astimezone(TZ)
        days_since_start = (now_msk().date() - start_dt.date()).days
        is_newbie = days_since_start <= 3

    today = now_msk().date()

    # --- последние 3 суток для советов ---
    three_days_ago = today - timedelta(days=3)
    adv_cutoff = datetime.combine(three_days_ago, time(0, 0), tzinfo=TZ).isoformat()

    cur.execute(
        """
        SELECT helped
        FROM advice_events
        WHERE user_id=? AND kind='advice' AND sent_at>=?
        """,
        (user_id, adv_cutoff),
    )
    rows = cur.fetchall()
    advice_count = len(rows)
    helped_advice_count = sum(1 for r in rows if r["helped"])

    help_rate: float | None
    if advice_count > 0:
        help_rate = helped_advice_count / advice_count
    else:
        help_rate = None

    # --- вчерашний чек-ин: heavy workload & low mood ---
    yesterday = today - timedelta(days=1)
    y_start = datetime.combine(yesterday, time(0, 0), tzinfo=TZ).isoformat()
    y_end = datetime.combine(today, time(0, 0), tzinfo=TZ).isoformat()

    cur.execute(
        """
        SELECT mood, workload
        FROM evening_checkins
        WHERE user_id=? AND sent_at>=? AND sent_at<? 
        ORDER BY sent_at DESC
        LIMIT 1
        """,
        (user_id, y_start, y_end),
    )
    row = cur.fetchone()
    if row is not None:
        yesterday_heavy_and_low = (row["workload"] >= 4) and (row["mood"] <= 3)
    else:
        yesterday_heavy_and_low = False

    # --- пропущенные чек-ины за последние 2 суток ---
    two_days_ago = today - timedelta(days=2)
    c_start = datetime.combine(two_days_ago, time(0, 0), tzinfo=TZ).isoformat()

    cur.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM evening_checkins
        WHERE user_id=? AND sent_at>=?
        """,
        (user_id, c_start),
    )
    row = cur.fetchone()
    missed_checkins_2days = (row["cnt"] == 0) if row is not None else True

    return ProbContext(
        is_newbie=is_newbie,
        help_rate=help_rate,
        advice_count=advice_count,
        helped_advice_count=helped_advice_count,
        yesterday_heavy_and_low=yesterday_heavy_and_low,
        missed_checkins_2days=missed_checkins_2days,
    )


def build_weekly_report(conn: sqlite3.Connection, user_id: int) -> str:
    """
    Мини-отчёт за последние 7 дней по пользователю.
    """
    now = now_msk()
    week_ago_iso = (now - timedelta(days=7)).isoformat()

    cur = conn.cursor()

    # Чек-ины
    cur.execute(
        """
        SELECT mood, workload, productivity
        FROM evening_checkins
        WHERE user_id=? AND sent_at>=?
        ORDER BY sent_at
        """,
        (user_id, week_ago_iso),
    )
    checkins = cur.fetchall()
    checkin_count = len(checkins)

    if checkin_count:
        avg_mood = sum(r["mood"] for r in checkins) / checkin_count
        avg_workload = sum(r["workload"] for r in checkins) / checkin_count
        avg_prod = sum(r["productivity"] for r in checkins) / checkin_count
    else:
        avg_mood = avg_workload = avg_prod = None

    # Советы
    cur.execute(
        """
        SELECT helped, rest_category
        FROM advice_events
        WHERE user_id=? AND kind='advice' AND sent_at>=?
        """,
        (user_id, week_ago_iso),
    )
    advice_rows = cur.fetchall()
    total_advice = len(advice_rows)
    helped_count = sum(1 for r in advice_rows if r["helped"])

    cats = [r["rest_category"] for r in advice_rows if r["rest_category"]]
    cat_label = None
    if cats:
        top_cat, _ = Counter(cats).most_common(1)[0]
        if top_cat == "active":
            cat_label = "активный отдых (прогулка/спорт)"
        elif top_cat == "passive":
            cat_label = "пассивный отдых (сон/соцсети)"
        elif top_cat == "social":
            cat_label = "социальный отдых (общение)"
        else:
            cat_label = "другое"

    lines: list[str] = []
    lines.append("Твой мини-отчёт за последние 7 дней.\n")

    lines.append(f"• Вечерние отчеты: {checkin_count} из 7 возможных.")
    if avg_mood is not None:
        lines.append(f"• Среднее настроение: {avg_mood:.1f}/5.")
        lines.append(f"• Средняя нагрузка: {avg_workload:.1f}/5.")
        lines.append(f"• Средняя продуктивность: {avg_prod:.1f}/5.")
    else:
        lines.append("• На этой неделе не было вечерних отчетов.")

    lines.append("")
    lines.append(f"• Совет получен {total_advice} раз(а).")
    lines.append(f"• После советов пауза помогла {helped_count} раз(а).")

    if cat_label:
        lines.append(f"• Чаще всего помогал: {cat_label}.")

    return "\n".join(lines)


# =========================
# JobQueue: global and personal tasks
# =========================

async def morning_broadcast(context: ContextTypes.DEFAULT_TYPE):
    """
    Утро (08:30 MSK) — поддержка:
      - сначала [support/morning],
      - если нет контента — [support/any].
    Для каждого пользователя:
      - выбираем текст;
      - отправляем;
      - логируем в advice_events (kind='support', time_window='morning' или 'any').
    """
    conn = get_connection(DB_PATH)
    try:
        users = all_users(conn)
        if not users:
            return

        for u in users:
            chat_id = u["chat_id"]
            user_id = u["id"]

            # сначала пытаемся support/morning
            adv = pick_support_message(conn, time_window="morning")
            time_window = "morning"
            if adv is None:
                adv = pick_support_any(conn)
                time_window = "any"

            if adv is None:
                # вообще нет поддержек — молчим
                continue

            # логируем событие
            # логируем событие и получаем id для фидбека
            event_id = create_advice_event(
                conn,
                user_id,
                advice_id=int(adv["id"]),
                kind="support",
                time_window=time_window,
                sent_at=now_msk(),
            )

            # отправляем сообщение с кнопками фидбека
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=adv["text"],
                    reply_markup=kb_advice_feedback(event_id),
                )
            except Exception:
                # В проде здесь логгер; пока просто игнорируем ошибки отправки.
                continue
    finally:
        conn.close()


async def day_broadcast(context: ContextTypes.DEFAULT_TYPE):
    """
    Дневной слот (14:00 MSK):
      - по каждому пользователю считаем p через ProbContext;
      - с вер. p отправляем advice/day;
      - иначе support (day или any);
      - для советов учитываем EXCLUDE_DAYS;
      - если нет подходящих советов — фолбэк на поддержку [support/any].
    """
    conn = get_connection(DB_PATH)
    try:
        users = all_users(conn)
        if not users:
            return

        for u in users:
            chat_id = u["chat_id"]
            user_id = u["id"]

            # собираем ProbContext из БД
            ctx = build_prob_context_for_user(conn, user_id)
            p = compute_day_advice_prob(ctx)
            kind_choice = choose_day_message_kind(p)  # "advice" или "support"

            adv_row: sqlite3.Row | None = None
            actual_kind: str
            actual_window: str

            if kind_choice == "advice":
                # пробуем совет для дневного окна с учётом EXCLUDE_DAYS
                adv_row = pick_advice_for_user(
                    conn,
                    user_id,
                    kind="advice",
                    time_window="day",
                    exclude_days=EXCLUDE_DAYS,
                )
                if adv_row is not None:
                    actual_kind = "advice"
                    actual_window = "day"
                else:
                    # нет новых советов — фолбэк на поддержку
                    adv_row = pick_support_any(conn)
                    if adv_row is None:
                        continue
                    actual_kind = "support"
                    actual_window = "any"
            else:
                # выбрали поддержку
                adv_row = pick_support_message(conn, time_window="day")
                if adv_row is not None:
                    actual_kind = "support"
                    actual_window = "day"
                else:
                    adv_row = pick_support_any(conn)
                    if adv_row is None:
                        continue
                    actual_kind = "support"
                    actual_window = "any"

            # логируем событие
            # логируем событие и получаем id
            event_id = create_advice_event(
                conn,
                user_id,
                advice_id=int(adv_row["id"]) if adv_row is not None else None,
                kind=actual_kind,
                time_window=actual_window,
                sent_at=now_msk(),
            )

            # отправляем сообщение с фидбек-кнопками
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=adv_row["text"],
                    reply_markup=kb_advice_feedback(event_id),
                )
            except Exception:
                continue
    finally:
        conn.close()


async def cmd_run_morning(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await morning_broadcast(context)


async def cmd_run_day(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await day_broadcast(context)


async def evening_broadcast(context: ContextTypes.DEFAULT_TYPE):
    """
    Вечернее напоминание (21:00 MSK):
      - пробегаем по всем пользователям;
      - если за сегодня у пользователя уже есть вечерний чек-ин — ничего не шлём;
      - иначе отправляем мягкое напоминание про /run_evening.
    """
    conn = get_connection(DB_PATH)
    try:
        users = all_users(conn)
        if not users:
            return

        today = now_msk().date()
        day_start = datetime.combine(today, time(0, 0), tzinfo=TZ).isoformat()
        day_end = datetime.combine(today + timedelta(days=1), time(0, 0), tzinfo=TZ).isoformat()

        cur = conn.cursor()

        for u in users:
            chat_id = u["chat_id"]
            user_id = u["id"]

            # Проверяем, есть ли уже чек-ин за сегодня
            cur.execute(
                """
                SELECT COUNT(*) AS cnt
                FROM evening_checkins
                WHERE user_id = ? AND sent_at >= ? AND sent_at < ?
                """,
                (user_id, day_start, day_end),
            )
            row = cur.fetchone()
            if row and row["cnt"] > 0:
                # Чек-ин уже есть, напоминание не шлём
                continue

            # Шлём напоминание
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    text=(
                        "Кажется, уже вечер.\n"
                        "Для подведения итогов дня — нажми /run_evening."
                    ),
                )
            except Exception:
                # В продакшене здесь лучше логировать
                continue
    finally:
        conn.close()


async def weekly_report_job(context: ContextTypes.DEFAULT_TYPE):
    """
    Джоба раз в 7 дней:
      1) шлёт мини-отчёт;
      2) задаёт вопрос про осознанность (1–5) и ставит флаг ожидания ответа.
    """
    job = context.job
    data = job.data or {}
    user_id = data.get("user_id")
    if not user_id:
        return

    conn = get_connection(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT chat_id FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()
        if row is None:
            return
        chat_id = row["chat_id"]

        report_text = build_weekly_report(conn, user_id)
    finally:
        conn.close()

    # 1) отчёт
    # 1) отчёт
    await context.bot.send_message(chat_id=chat_id, text=report_text)

    # 2) вопрос про осознанность
    await context.bot.send_message(
        chat_id=chat_id,
        text=(
            "Теперь небольшой вопрос\n\n"
            "За последнюю неделю ты замечал, что тебе необходима пауза или перерыв?\n\n"
            "Выбери оценку от 1 до 5:\n"
            "1 — почти никогда не замечаю\n"
            "5 — почти всегда замечаю вовремя"
        ),
        reply_markup=kb_awareness_inline(),
    )


async def awareness_job(context: ContextTypes.DEFAULT_TYPE):
    return


async def personal_break_reminder(context: ContextTypes.DEFAULT_TYPE):
    """
    Персональное напоминание про 10 минут на завтра.
    Ожидаем в data: {"user_id": int, "remind_at": isoformat(str)}.
    """
    job = context.job
    data = job.data or {}
    user_id = data.get("user_id")
    remind_at_str = data.get("remind_at")

    if user_id is None:
        return

    conn = get_connection(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT chat_id FROM users WHERE id=?", (user_id,))
        row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        return

    chat_id = row["chat_id"]

    time_part = ""
    if remind_at_str:
        try:
            dt = datetime.fromisoformat(remind_at_str).astimezone(TZ)
            time_part = dt.strftime("%H:%M")
        except Exception:
            time_part = ""

    if time_part:
        text = f"Напоминание про твои 10 минут для себя в {time_part}."
    else:
        text = "Напоминание про твои 10 минут для себя."

    await context.bot.send_message(chat_id=chat_id, text=text)


def schedule_global_jobs(job_queue: JobQueue):
    job_queue.run_daily(
        morning_broadcast,
        time=MORNING_TIME,
        name="morning_broadcast",
    )
    job_queue.run_daily(
        day_broadcast,
        time=DAY_TIME,
        name="day_broadcast",
    )
    job_queue.run_daily(
        evening_broadcast,
        time=EVENING_TIME,
        name="evening_broadcast",
    )


def schedule_user_personal_jobs(job_queue: JobQueue, user_id: int, start_date: datetime):
    start_local = start_date.astimezone(TZ)
    first_day = (start_local + timedelta(days=7)).date()

    first_report_at = datetime.combine(
        first_day,
        time(WEEKLY_REPORT_HOUR, 0),
        tzinfo=TZ,
    )

    job_queue.run_repeating(
        weekly_report_job,
        interval=timedelta(days=7),
        first=first_report_at,
        name=f"weekly_report_{user_id}",
        data={"user_id": user_id},
    )


# =========================
# Rest categories & validation
# =========================

def rest_category_of(rest_type: str) -> str:
    """
    Маппинг типа отдыха в категорию:
      - Прогулка, Спорт, Друзья -> active
      - Сон -> passive
      - Соцсети -> passive (условно «пассивное потребление»)
      - Другое… -> other
    Любой незнакомый тип -> other.
    """
    if not rest_type:
        return "other"

    t = rest_type.strip().lower()
    if t in ("прогулка", "прогулка ", "walk"):
        return "active"
    if t in ("спорт", "sport"):
        return "active"
    if t in ("друзья", "friends"):
        return "social"
    if t in ("сон", "sleep"):
        return "passive"
    if t in ("соцсети", "соц сети", "social media"):
        return "passive"
    if "другое" in t:
        return "other"
    return "other"


def is_valid_note(text: str) -> bool:
    """
    Вечерняя заметка:
      - непустой текст
      - длина ≤ 1000
    """
    if text is None:
        return False
    s = text.strip()
    return bool(s) and len(s) <= 1000


def is_valid_custom_rest_note(text: str) -> bool:
    """
    Текст для 'Другое…':
      - 1..200 символов (после трима)
    """
    if text is None:
        return False
    s = text.strip()
    return 1 <= len(s) <= 200


def parse_planned_time_input(raw: str) -> time | None:
    """
    Парсинг ввода ЧЧ:ММ для планирования 10 минут завтра.
    Валидное время:
      - 'HH:MM', 00<=HH<=23, 00<=MM<=59
    При ошибке вернуть None.
    """
    if not raw:
        return None
    s = raw.strip()
    parts = s.split(":")
    if len(parts) != 2:
        return None
    hh, mm = parts
    if not (hh.isdigit() and mm.isdigit()):
        return None
    h = int(hh)
    m = int(mm)
    if not (0 <= h <= 23 and 0 <= m <= 59):
        return None
    return time(h, m)


# =========================
# Evening check-in state machine (pure logic)
# =========================

@dataclass
class CheckinSession:
    """
    Чистое представление одной сессии вечернего чек-ина.
    Используется в тестах и бизнес-логике; Telegram-обёртка появится позже.
    """
    user_id: int
    sent_at: datetime
    step: str  # "MOOD" | "WORKLOAD" | "PRODUCTIVITY" | "NOTE_CHOICE" | "NOTE_TEXT" | "PLAN_CHOICE" | "PLAN_TIME" | "DONE"

    mood: int | None = None
    workload: int | None = None
    productivity: int | None = None
    note: str | None = None
    plan_scheduled: bool = False  # защита от двойного планирования внутри сессии


def save_checkin_to_db(session: CheckinSession) -> int:
    """
    Сохранить завершённый чек-ин в таблицу evening_checkins.
    Возвращает id вставленной строки.
    """
    conn = get_connection(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO evening_checkins (user_id, sent_at, mood, workload, productivity, note)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session.user_id,
                session.sent_at.isoformat(),
                session.mood,
                session.workload,
                session.productivity,
                session.note,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


# Telegram Conversation states for evening check-in
(
    CHECKIN_MOOD,
    CHECKIN_WORKLOAD,
    CHECKIN_PRODUCTIVITY,
    CHECKIN_NOTE_CHOICE,
    CHECKIN_NOTE_TEXT,
    CHECKIN_PLAN_CHOICE,
    CHECKIN_PLAN_TIME,
) = range(7)


def start_checkin_session(user_id: int, *, when: datetime | None = None) -> CheckinSession:
    """
    Старт вечернего чек-ина: первый шаг — выбор настроения (mood).
    """
    if when is None:
        when = now_msk()
    return CheckinSession(
        user_id=user_id,
        sent_at=when,
        step="MOOD",
    )


def _validate_1_5(value: int) -> bool:
    return isinstance(value, int) and 1 <= value <= 5


def checkin_set_mood(session: CheckinSession, mood: int) -> tuple[CheckinSession, str | None]:
    if session.step != "MOOD":
        return session, "invalid_state"
    if not _validate_1_5(mood):
        return session, "invalid_value"
    session.mood = mood
    session.step = "WORKLOAD"
    return session, None


def checkin_set_workload(session: CheckinSession, workload: int) -> tuple[CheckinSession, str | None]:
    if session.step != "WORKLOAD":
        return session, "invalid_state"
    if not _validate_1_5(workload):
        return session, "invalid_value"
    session.workload = workload
    session.step = "PRODUCTIVITY"
    return session, None


def checkin_set_productivity(session: CheckinSession, productivity: int) -> tuple[CheckinSession, str | None]:
    if session.step != "PRODUCTIVITY":
        return session, "invalid_state"
    if not _validate_1_5(productivity):
        return session, "invalid_value"
    session.productivity = productivity
    session.step = "NOTE_CHOICE"
    return session, None


def checkin_choose_note(session: CheckinSession, want_note: bool) -> tuple[CheckinSession, str | None]:
    """
    Выбор: 'Оставим заметку?' Да/Нет.
    """
    if session.step != "NOTE_CHOICE":
        return session, "invalid_state"
    if want_note:
        session.step = "NOTE_TEXT"
    else:
        session.step = "PLAN_CHOICE"
    return session, None


def checkin_set_note_text(session: CheckinSession, text: str | None, *, skip: bool = False) -> tuple[
    CheckinSession, str | None]:
    """
    Установка текста заметки:
      - если skip=True → пропускаем без сохранения текста;
      - иначе требуем валидный текст is_valid_note.
    """
    if session.step != "NOTE_TEXT":
        return session, "invalid_state"

    if skip:
        # пропуск заметки
        session.note = None
        session.step = "PLAN_CHOICE"
        return session, None

    if not is_valid_note(text or ""):
        return session, "invalid_text"

    session.note = text.strip()
    session.step = "PLAN_CHOICE"
    return session, None


def checkin_choose_plan(session: CheckinSession, want_plan: bool) -> tuple[CheckinSession, str | None]:
    """
    'Запланировать 10 минут завтра?' Да/Нет.
    """
    # Если уже успели запланировать в этой сессии — любое повторное "давай спланируем"
    # трактуем как защиту от двойного планирования.
    if session.plan_scheduled:
        return session, "already_planned"

    if session.step not in ("PLAN_CHOICE", "PLAN_TIME"):
        return session, "invalid_state"

    if not want_plan:
        session.step = "DONE"
        return session, None

    # хотим планировать
    session.step = "PLAN_TIME"
    return session, None


def checkin_set_plan_time(session: CheckinSession, time_input: str) -> tuple[CheckinSession, str | None]:
    """
    Ввод времени (кнопка '16:00' или ручной ввод 'ЧЧ:ММ'):
      - валидируем через parse_planned_time_input;
      - при успехе отмечаем plan_scheduled=True и завершаем сессию (DONE);
      - при повторной попытке — ошибка 'already_planned'.
    """
    if session.step != "PLAN_TIME":
        return session, "invalid_state"

    if session.plan_scheduled:
        return session, "already_planned"

    t = parse_planned_time_input(time_input)
    if t is None:
        return session, "invalid_time"

    # тут не сохраняем конкретное время в БД — эта логика будет на уровне Telegram/JobQueue
    session.plan_scheduled = True
    session.step = "DONE"
    return session, None


# =========================
# Telegram keyboards (UI helpers)
# =========================

def kb_1_5() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [["1", "2", "3", "4", "5"]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def kb_yes_no() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [["Да", "Нет"]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def kb_plan_times() -> ReplyKeyboardMarkup:
    return ReplyKeyboardMarkup(
        [["12:00", "14:00", "16:00"]],
        resize_keyboard=True,
        one_time_keyboard=True,
    )


def kb_advice_feedback(advice_event_id: int) -> InlineKeyboardMarkup:
    buttons = [
        [
            InlineKeyboardButton("1", callback_data=f"advfb:rate:{advice_event_id}:1"),
            InlineKeyboardButton("2", callback_data=f"advfb:rate:{advice_event_id}:2"),
            InlineKeyboardButton("3", callback_data=f"advfb:rate:{advice_event_id}:3"),
            InlineKeyboardButton("4", callback_data=f"advfb:rate:{advice_event_id}:4"),
            InlineKeyboardButton("5", callback_data=f"advfb:rate:{advice_event_id}:5"),
        ],
        [
            InlineKeyboardButton(
                "Сделал паузу — помогло",
                callback_data=f"advfb:helped:{advice_event_id}",
            )
        ],
    ]
    return InlineKeyboardMarkup(buttons)


def kb_awareness_inline() -> InlineKeyboardMarkup:
    """
    Инлайн-кнопки 1–5 под вопросом про осознанность.
    """
    buttons = [
        [
            InlineKeyboardButton("1", callback_data="aware:1"),
            InlineKeyboardButton("2", callback_data="aware:2"),
            InlineKeyboardButton("3", callback_data="aware:3"),
            InlineKeyboardButton("4", callback_data="aware:4"),
            InlineKeyboardButton("5", callback_data="aware:5"),
        ]
    ]
    return InlineKeyboardMarkup(buttons)


def kb_rest_types(advice_event_id: int) -> InlineKeyboardMarkup:
    """
    Выбор типа отдыха после 'Сделал паузу — помогло'.
    """
    buttons = [
        [
            InlineKeyboardButton("Прогулка", callback_data=f"advfb:rest:{advice_event_id}:walk"),
            InlineKeyboardButton("Спорт", callback_data=f"advfb:rest:{advice_event_id}:sport"),
        ],
        [
            InlineKeyboardButton("Друзья", callback_data=f"advfb:rest:{advice_event_id}:friends"),
            InlineKeyboardButton("Сон", callback_data=f"advfb:rest:{advice_event_id}:sleep"),
        ],
        [
            InlineKeyboardButton("Соцсети", callback_data=f"advfb:rest:{advice_event_id}:social"),
        ],
        [
            InlineKeyboardButton("Другое…", callback_data=f"advfb:rest:{advice_event_id}:other"),
            InlineKeyboardButton("Пропустить", callback_data=f"advfb:rest:{advice_event_id}:skip"),
        ],
    ]
    return InlineKeyboardMarkup(buttons)


# =========================
# Advice feedback state machine (pure logic)
# =========================

@dataclass
class AdviceFeedbackSession:
    """
    Состояние флоу после отправки совета/поддержки.
    """
    user_id: int
    advice_event_id: int
    step: str  # "INITIAL" | "REST_TYPE" | "CUSTOM_REST_NOTE" | "DONE"

    rating: int | None = None
    helped: bool = False
    not_relevant: bool = False
    rest_type: str | None = None
    rest_note: str | None = None


def advice_set_rating(session: AdviceFeedbackSession, rating: int) -> tuple[AdviceFeedbackSession, str | None]:
    """
    Оценка 1..5. При установке сбрасываем not_relevant в 0.
    """
    if session.step not in ("INITIAL", "DONE"):
        return session, "invalid_state"
    if not _validate_1_5(rating):
        return session, "invalid_value"
    session.rating = rating
    session.not_relevant = False
    # остаёмся в INITIAL, пока пользователь не нажмёт другие кнопки
    return session, None


def advice_mark_helped(session: AdviceFeedbackSession) -> tuple[AdviceFeedbackSession, str | None]:
    """
    'Сделал паузу — помогло': ставим helped=1 и переходим к выбору типа отдыха.
    """
    if session.step not in ("INITIAL", "REST_TYPE", "CUSTOM_REST_NOTE"):
        return session, "invalid_state"
    session.helped = True
    session.step = "REST_TYPE"
    return session, None


def advice_choose_rest_type(session: AdviceFeedbackSession, rest_type: str) -> tuple[AdviceFeedbackSession, str | None]:
    """
    Выбор 'Прогулка/Спорт/Сон/Соцсети/Друзья/Другое…/Пропустить'.
    """
    if session.step != "REST_TYPE":
        return session, "invalid_state"

    label = rest_type.strip().lower()
    if "пропустить" in label:
        # пользователь не хочет уточнять тип отдыха
        session.step = "DONE"
        return session, None

    session.rest_type = rest_type

    if "другое" in label:
        # нужен дополнительный текст или явный пропуск
        session.step = "CUSTOM_REST_NOTE"
        return session, None

    # обычный тип отдыха, можем закончить
    session.step = "DONE"
    return session, None


def advice_set_custom_rest_note(
        session: AdviceFeedbackSession,
        text: str | None,
        *,
        skip: bool = False,
) -> tuple[AdviceFeedbackSession, str | None]:
    """
    Обработка текста для 'Другое…':
      - если skip=True → выходим без текста;
      - иначе требуется непустой текст 1..200 символов.
    """
    if session.step != "CUSTOM_REST_NOTE":
        return session, "invalid_state"

    if skip:
        session.rest_note = None
        session.step = "DONE"
        return session, None

    if not is_valid_custom_rest_note(text or ""):
        return session, "invalid_text"

    session.rest_note = text.strip()
    session.step = "DONE"
    return session, None


def advice_mark_not_relevant(session: AdviceFeedbackSession) -> tuple[AdviceFeedbackSession, str | None]:
    """
    'Не в тему': not_relevant=1, но если helped=1 — игнорируем (по ТЗ).
    """
    if session.helped:
        # игнор, ничего не меняем
        return session, None

    if session.step not in ("INITIAL", "REST_TYPE", "CUSTOM_REST_NOTE", "DONE"):
        return session, "invalid_state"

    session.not_relevant = True
    return session, None


def save_advice_feedback_to_db(session: AdviceFeedbackSession) -> None:
    """
    Обновить запись advice_events по advice_event_id из session.
    Пишем:
      - rating (1..5 или NULL)
      - helped (0/1)
      - not_relevant (0/1)
      - rest_type
      - rest_category (из rest_category_of)
      - rest_note
    """
    conn = get_connection(DB_PATH)
    try:
        cur = conn.cursor()

        rest_type = session.rest_type
        rest_category = rest_category_of(rest_type) if rest_type else None

        cur.execute(
            """
            UPDATE advice_events
            SET
              rating = ?,
              helped = ?,
              not_relevant = ?,
              rest_type = ?,
              rest_category = ?,
              rest_note = ?
            WHERE id = ?
            """,
            (
                session.rating,
                1 if session.helped else 0,
                1 if session.not_relevant else 0,
                rest_type,
                rest_category,
                session.rest_note,
                session.advice_event_id,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# =========================
# Telegram handlers: evening check-in
# =========================

CHECKIN_SESSION_KEY = "checkin_session"
ADVICE_FEEDBACK_SESSION_KEY = "advice_feedback_session"


async def cmd_run_evening(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """
    Сервисная команда /run_evening — запускает вечерний чек-ин для инициатора.
    В проде авто-чек-ин будет идти из evening_broadcast.
    """
    chat = update.effective_chat
    if chat is None:
        return ConversationHandler.END

    conn = get_connection(DB_PATH)
    try:
        user_id = get_or_create_user(conn, chat.id)
    finally:
        conn.close()

    session = start_checkin_session(user_id)
    context.user_data[CHECKIN_SESSION_KEY] = session

    await update.message.reply_text(
        "Как у тебя настроение сегодня по шкале от 1 до 5?",
        reply_markup=kb_1_5(),
    )
    return CHECKIN_MOOD


async def checkin_mood_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session: CheckinSession = context.user_data.get(CHECKIN_SESSION_KEY)
    if session is None:
        await update.message.reply_text("Что-то пошло не так, попробуй ещё раз позже.")
        return ConversationHandler.END

    text = (update.message.text or "").strip()
    if not text.isdigit():
        await update.message.reply_text("Пожалуйста, отправь число от 1 до 5.", reply_markup=kb_1_5())
        return CHECKIN_MOOD

    mood = int(text)
    session, err = checkin_set_mood(session, mood)
    if err:
        await update.message.reply_text("Пожалуйста, отправь число от 1 до 5.", reply_markup=kb_1_5())
        return CHECKIN_MOOD

    context.user_data[CHECKIN_SESSION_KEY] = session
    await update.message.reply_text(
        "А как была нагрузка сегодня (1 — почти не было, 5 — очень много)?",
        reply_markup=kb_1_5(),
    )
    return CHECKIN_WORKLOAD


async def checkin_workload_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session: CheckinSession = context.user_data.get(CHECKIN_SESSION_KEY)
    if session is None:
        await update.message.reply_text("Что-то пошло не так, попробуй ещё раз позже.")
        return ConversationHandler.END

    text = (update.message.text or "").strip()
    if not text.isdigit():
        await update.message.reply_text("Пожалуйста, отправь число от 1 до 5.", reply_markup=kb_1_5())
        return CHECKIN_WORKLOAD

    val = int(text)
    session, err = checkin_set_workload(session, val)
    if err:
        await update.message.reply_text("Пожалуйста, отправь число от 1 до 5.", reply_markup=kb_1_5())
        return CHECKIN_WORKLOAD

    context.user_data[CHECKIN_SESSION_KEY] = session
    await update.message.reply_text(
        "Насколько ты доволен(а) своей продуктивностью сегодня (1–5)?",
        reply_markup=kb_1_5(),
    )
    return CHECKIN_PRODUCTIVITY


async def checkin_productivity_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session: CheckinSession = context.user_data.get(CHECKIN_SESSION_KEY)
    if session is None:
        await update.message.reply_text("Что-то пошло не так, попробуй ещё раз позже.")
        return ConversationHandler.END

    text = (update.message.text or "").strip()
    if not text.isdigit():
        await update.message.reply_text("Пожалуйста, отправь число от 1 до 5.", reply_markup=kb_1_5())
        return CHECKIN_PRODUCTIVITY

    val = int(text)
    session, err = checkin_set_productivity(session, val)
    if err:
        await update.message.reply_text("Пожалуйста, отправь число от 1 до 5.", reply_markup=kb_1_5())
        return CHECKIN_PRODUCTIVITY

    context.user_data[CHECKIN_SESSION_KEY] = session
    await update.message.reply_text(
        "Оставим короткую заметку о дне?",
        reply_markup=kb_yes_no(),
    )
    return CHECKIN_NOTE_CHOICE


async def checkin_note_choice_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session: CheckinSession = context.user_data.get(CHECKIN_SESSION_KEY)
    if session is None:
        await update.message.reply_text("Что-то пошло не так, попробуй ещё раз позже.")
        return ConversationHandler.END

    text = (update.message.text or "").strip().lower()
    want_note = text == "да"

    session, err = checkin_choose_note(session, want_note=want_note)
    if err:
        await update.message.reply_text("Пожалуйста, выбери: Да или Нет.", reply_markup=kb_yes_no())
        return CHECKIN_NOTE_CHOICE

    context.user_data[CHECKIN_SESSION_KEY] = session

    if want_note:
        await update.message.reply_text(
            "Напиши пару слов о своём дне. Если передумал(а) — напиши «Пропустить».",
            reply_markup=ReplyKeyboardRemove(),
        )
        return CHECKIN_NOTE_TEXT
    else:
        await update.message.reply_text(
            "Запланируем 10 минут для себя на завтра?",
            reply_markup=kb_yes_no(),
        )
        return CHECKIN_PLAN_CHOICE


async def checkin_note_text_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session: CheckinSession = context.user_data.get(CHECKIN_SESSION_KEY)
    if session is None:
        await update.message.reply_text("Что-то пошло не так, попробуй ещё раз позже.")
        return ConversationHandler.END

    text = (update.message.text or "").strip()
    if text.lower() == "пропустить":
        session, err = checkin_set_note_text(session, None, skip=True)
    else:
        session, err = checkin_set_note_text(session, text, skip=False)

    if err == "invalid_text":
        await update.message.reply_text(
            "Заметка должна быть не пустой и не длиннее 1000 символов. "
            "Можешь написать ещё раз или отправить «Пропустить».",
        )
        return CHECKIN_NOTE_TEXT
    if err:
        await update.message.reply_text("Что-то пошло не так, попробуй ещё раз позже.")
        return ConversationHandler.END

    context.user_data[CHECKIN_SESSION_KEY] = session
    await update.message.reply_text(
        "Запланируем 10 минут для себя на завтра?",
        reply_markup=kb_yes_no(),
    )
    return CHECKIN_PLAN_CHOICE


async def checkin_plan_choice_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session: CheckinSession = context.user_data.get(CHECKIN_SESSION_KEY)
    if session is None:
        await update.message.reply_text("Что-то пошло не так, попробуй ещё раз позже.")
        return ConversationHandler.END

    text = (update.message.text or "").strip().lower()
    want_plan = text == "да"

    session, err = checkin_choose_plan(session, want_plan=want_plan)
    if err == "already_planned":
        await update.message.reply_text(
            "У тебя уже запланированы 10 минут в этой сессии.",
            reply_markup=ReplyKeyboardRemove(),
        )
        save_checkin_to_db(session)
        context.user_data.pop(CHECKIN_SESSION_KEY, None)
        return ConversationHandler.END
    if err:
        await update.message.reply_text("Пожалуйста, выбери: Да или Нет.", reply_markup=kb_yes_no())
        return CHECKIN_PLAN_CHOICE

    context.user_data[CHECKIN_SESSION_KEY] = session

    if not want_plan:
        # завершаем без планирования
        save_checkin_to_db(session)
        context.user_data.pop(CHECKIN_SESSION_KEY, None)
        await update.message.reply_text(
            "Спасибо, чек-ин сохранён. Хорошего вечера!",
            reply_markup=ReplyKeyboardRemove(),
        )
        return ConversationHandler.END

    # переходим к выбору времени
    await update.message.reply_text(
        "Во сколько завтра напомнить? Можно выбрать 12:00, 14:00 или 16:00, "
        "или написать своё время в формате ЧЧ:ММ.",
        reply_markup=kb_plan_times(),
    )
    return CHECKIN_PLAN_TIME


async def checkin_plan_time_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    session: CheckinSession = context.user_data.get(CHECKIN_SESSION_KEY)
    if session is None:
        await update.message.reply_text("Что-то пошло не так, попробуй ещё раз позже.")
        return ConversationHandler.END

    text = (update.message.text or "").strip()

    # сначала прогоняем через чистую логику
    session, err = checkin_set_plan_time(session, text)
    if err == "invalid_time":
        await update.message.reply_text(
            "Время нужно в формате ЧЧ:ММ, например 16:00. Попробуй ещё раз.",
            reply_markup=kb_plan_times(),
        )
        return CHECKIN_PLAN_TIME
    if err == "already_planned":
        await update.message.reply_text(
            "У тебя уже есть запланированное время на завтра в этой сессии.",
            reply_markup=ReplyKeyboardRemove(),
        )
        save_checkin_to_db(session)
        context.user_data.pop(CHECKIN_SESSION_KEY, None)
        return ConversationHandler.END
    if err:
        await update.message.reply_text("Что-то пошло не так, попробуй ещё раз позже.")
        return ConversationHandler.END

    # Успех: session.plan_scheduled=True, step="DONE".
    # Повторно парсим ввод, чтобы получить объект time.
    t = parse_planned_time_input(text)
    if t is None:
        # Теоретически не должно случиться, но подстрахуемся:
        save_checkin_to_db(session)
        context.user_data.pop(CHECKIN_SESSION_KEY, None)
        await update.message.reply_text(
            "Готово, 10 минут на завтра запланированы. Спасибо за уделенное время!",
            reply_markup=ReplyKeyboardRemove(),
        )
        return ConversationHandler.END

    # Дата «завтра» относительно sent_at сессии.
    base_dt = session.sent_at.astimezone(TZ)
    tomorrow_date = (base_dt + timedelta(days=1)).date()
    remind_at = datetime.combine(tomorrow_date, t, tzinfo=TZ)

    # Сохраняем чек-ин и план в БД.
    checkin_id = save_checkin_to_db(session)

    conn = get_connection(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO planned_breaks (user_id, remind_at, created_at)
            VALUES (?, ?, ?)
            """,
            (
                session.user_id,
                remind_at.isoformat(),
                now_msk().isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()

    # Ставит персональный JobQueue-джоб.
    context.application.job_queue.run_once(
        personal_break_reminder,
        when=remind_at,
        name=f"personal_break_{session.user_id}_{remind_at.isoformat()}",
        data={"user_id": session.user_id, "remind_at": remind_at.isoformat()},
    )

    context.user_data.pop(CHECKIN_SESSION_KEY, None)

    # Красивый текст с конкретным временем
    time_str = remind_at.astimezone(TZ).strftime("%H:%M")
    await update.message.reply_text(
        f"Готово, 10 минут на завтра запланированы на {time_str}. Спасибо за уделенное время!",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


async def advice_feedback_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработка инлайн-кнопок под советом/поддержкой.
    Формат callback_data:
      - advfb:rate:<event_id>:<1-5>
      - advfb:helped:<event_id>
      - advfb:notrel:<event_id>
      - advfb:rest:<event_id>:<code>
        где code: walk|sport|friends|sleep|social|other|skip
    """
    query = update.callback_query
    if query is None or not query.data:
        return

    data = query.data
    parts = data.split(":")
    if len(parts) < 3 or parts[0] != "advfb":
        return

    action = parts[1]
    try:
        event_id = int(parts[2])
    except ValueError:
        return

    # достаём user_id из advice_events
    conn = get_connection(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute("SELECT user_id FROM advice_events WHERE id=?", (event_id,))
        row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        await query.answer("Не нашла это событие, попробуй позже.")
        return

    db_user_id = int(row["user_id"])

    # берём/создаём сессию фидбека
    session: AdviceFeedbackSession | None = context.user_data.get(ADVICE_FEEDBACK_SESSION_KEY)
    if session is None or session.advice_event_id != event_id:
        session = AdviceFeedbackSession(
            user_id=db_user_id,
            advice_event_id=event_id,
            step="INITIAL",
        )

    # ----- обработка действий -----
    if action == "rate":
        if len(parts) < 4:
            await query.answer("Ошибка рейтинга.")
            return
        try:
            rating = int(parts[3])
        except ValueError:
            await query.answer("Ошибка рейтинга.")
            return

        session, err = advice_set_rating(session, rating)
        if err:
            await query.answer("Не удалось сохранить рейтинг.")
        else:
            save_advice_feedback_to_db(session)
            context.user_data[ADVICE_FEEDBACK_SESSION_KEY] = session

            # убираем кнопки оценок
            try:
                await query.edit_message_reply_markup(reply_markup=None)
            except Exception:
                pass

            await query.answer(f"Спасибо за оценку: {rating}/5!")

        return

    if action == "helped":
        session, err = advice_mark_helped(session)
        if err:
            await query.answer("Не получилось отметить паузу.")
            return

        save_advice_feedback_to_db(session)
        context.user_data[ADVICE_FEEDBACK_SESSION_KEY] = session

        # убираем старые кнопки (1–5 и «Сделал паузу»)
        try:
            await query.edit_message_reply_markup(reply_markup=None)
        except Exception:
            pass

        # Переход к выбору типа отдыха
        await query.answer("Супер, что помогло!")
        await query.message.reply_text(
            "Что ты сделал(а) в паузе?",
            reply_markup=kb_rest_types(event_id),
        )
        return

    if action == "rest":
        if len(parts) < 4:
            await query.answer("Ошибка выбора отдыха.")
            return
        code = parts[3]

        # маппинг к человеку читаемым лейблам
        if code == "walk":
            label = "Прогулка"
        elif code == "sport":
            label = "Спорт"
        elif code == "friends":
            label = "Друзья"
        elif code == "sleep":
            label = "Сон"
        elif code == "social":
            label = "Соцсети"
        elif code == "other":
            label = "Другое…"
        elif code == "skip":
            label = "Пропустить"
        else:
            await query.answer("Неизвестный тип отдыха.")
            return

        session, err = advice_choose_rest_type(session, label)
        if err:
            await query.answer("Не удалось сохранить тип отдыха.")
            return

        # Если пользователь нажал "Пропустить" или любой стандартный тип → можно сохранять и завершать.
        if session.step == "DONE":
            save_advice_feedback_to_db(session)
            context.user_data[ADVICE_FEEDBACK_SESSION_KEY] = session

            # УДАЛЯЕМ сообщение «Что ты сделал(а) в паузе?» вместе с кнопками
            try:
                await query.message.delete()
            except Exception:
                pass

            if label == "Пропустить":
                await query.answer("Хорошо, без уточнения.")
            else:
                await query.answer("Записала, чем ты отдыхал(а).")
            return

        # Если "Другое…" → ждём текст
        if session.step == "CUSTOM_REST_NOTE":
            context.user_data[ADVICE_FEEDBACK_SESSION_KEY] = session

            # УДАЛЯЕМ сообщение с кнопками выбора отдыха
            try:
                await query.message.delete()
            except Exception:
                pass

            await query.answer()
            await query.message.reply_text(
                "Напиши пару слов, что это был за отдых. "
                "Если не хочешь уточнять — отправь «Пропустить».",
                reply_markup=ReplyKeyboardRemove(),
            )
            return

    # на всякий случай
    await query.answer("Что-то пошло не так.")


async def advice_custom_rest_note_step(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработка текстового ответа после 'Другое…' в типе отдыха.
    Работает только если в user_data лежит AdviceFeedbackSession с step='CUSTOM_REST_NOTE'.
    """
    session: AdviceFeedbackSession | None = context.user_data.get(ADVICE_FEEDBACK_SESSION_KEY)
    if session is None or session.step != "CUSTOM_REST_NOTE":
        # Это сообщение не про фидбек, игнорируем.
        return

    text = (update.message.text or "").strip()

    if text.lower() == "пропустить":
        session, err = advice_set_custom_rest_note(session, None, skip=True)
    else:
        session, err = advice_set_custom_rest_note(session, text, skip=False)

    if err == "invalid_text":
        await update.message.reply_text(
            "Описание должно быть от 1 до 200 символов. "
            "Попробуй ещё раз или отправь «Пропустить».",
        )
        return
    if err:
        await update.message.reply_text("Не удалось сохранить, попробуй ещё раз позже.")
        return

    # Успех
    save_advice_feedback_to_db(session)
    context.user_data[ADVICE_FEEDBACK_SESSION_KEY] = session
    await update.message.reply_text("Тип отдыха сохранён 🙂")


async def awareness_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Обработка инлайн-кнопок 1–5 под вопросом про осознанность.
    Формат callback_data: 'aware:<1-5>'.
    """
    query = update.callback_query
    if query is None or not query.data:
        return

    data = query.data
    if not data.startswith("aware:"):
        return

    # парсим оценку
    try:
        score = int(data.split(":")[1])
    except ValueError:
        await query.answer("Ошибка оценки.")
        return

    if not 1 <= score <= 5:
        await query.answer("Оценка должна быть от 1 до 5.")
        return

    chat = query.message.chat
    if chat is None:
        return
    chat_id = chat.id

    # проверяем флаг ожидания (как в weekly_report_job)

    # сохраняем в БД
    conn = get_connection(DB_PATH)
    try:
        user_id = get_or_create_user(conn, chat_id)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO awareness (user_id, sent_at, response)
            VALUES (?, ?, ?)
            """,
            (user_id, now_msk().isoformat(), str(score)),
        )
        conn.commit()
    finally:
        conn.close()

    # убираем кнопки под вопросом
    try:
        await query.edit_message_reply_markup(reply_markup=None)
    except Exception:
        pass

    await query.answer("Спасибо за ответ!")

    # мотивационная мини-фраза
    phrases = [
        "Спасибо, что уделяешь внимание себе — это уже большая работа.",
        "Круто, что ты отслеживаешь своё состояние. Это основа баланса.",
        "Отмечать такие моменты — первый шаг к более бережному режиму.",
        "Спасибо за ответ. Ты реально молодец, что до этого дошёл(ла).",
    ]
    msg = random.choice(phrases)

    await context.bot.send_message(chat_id=chat_id, text=msg)


async def cmd_run_report(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Сервисная команда для ручного запуска отчёта и вопроса про осознанность.
    Удобно для тестов, не дожидаясь 7 дней.
    """
    chat = update.effective_chat
    if chat is None:
        return

    conn = get_connection(DB_PATH)
    try:
        user_id = get_or_create_user(conn, chat.id)
        report_text = build_weekly_report(conn, user_id)
    finally:
        conn.close()

    # отчёт
    await update.message.reply_text(report_text)

    # вопрос + флаг ожидания
    await update.message.reply_text(
        "Теперь небольшой вопрос\n\n"
        "За последнюю неделю ты замечал, что тебе необходима пауза или перерыв?\n\n"
        "Выбери оценку от 1 до 5:\n"
        "1 — почти никогда не замечаю\n"
        "5 — почти всегда замечаю вовремя",
        reply_markup=kb_awareness_inline(),
    )


async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """
    /start: регистрация пользователя и краткое объяснение механики.
    """
    chat = update.effective_chat
    if chat is None:
        return

    conn = get_connection(DB_PATH)
    try:
        user_id = get_or_create_user(conn, chat.id)
        cur = conn.cursor()
        cur.execute("SELECT start_date FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
        start_date_str = row["start_date"]
    finally:
        conn.close()

    # Планируем персональные задачи T+7
    start_dt = datetime.fromisoformat(start_date_str)
    schedule_user_personal_jobs(context.application.job_queue, user_id, start_dt)

    text = (
        "Привет! Я помогу тебе отслеживать нагрузку и отдых.\n\n"
        "• Утром и днём я буду присылать короткие сообщения — поддержку и советы.\n"
        "• Вечером я сам спрошу, как прошёл день.\n"
        "• После заметки предложу запланировать 10 минут на завтра.\n"
        "• Через 7 дней пришлю мини-отчёт и задам вопрос про изменения.\n\n"
    )

    await update.message.reply_text(text, reply_markup=ReplyKeyboardRemove())


# =========================
# App init & run
# =========================

def init_app(db_path: Path = DB_PATH, content_path: Path = CONTENT_PATH) -> None:
    """
    Инициализация БД и загрузка контента, если advices пусты.
    Используется и в тестах (через части) и при запуске бота.
    """
    conn = get_connection(db_path)
    try:
        init_db(conn)
        if advices_empty(conn):
            records = load_content_from_file(content_path)
            import_advices(conn, records, clear_existing=True)
    finally:
        conn.close()


def build_application() -> "object":
    """
    Собрать Telegram Application:
      - инициализировать БД/контент;
      - повесить handlers;
      - повесить глобальные джобы.
    """
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("Переменная окружения BOT_TOKEN не задана")

    init_app()

    defaults = Defaults(tzinfo=TZ)
    app = ApplicationBuilder().token(token).defaults(defaults).build()

    # Глобальные джобы (утро/день/вечер)
    schedule_global_jobs(app.job_queue)

    # Хэндлер /start
    app.add_handler(CommandHandler("start", cmd_start))

    # Хэндлер вечернего чек-ина (сервисная команда /run_evening)
    # Хэндлер вечернего чек-ина (сервисная команда /run_evening)
    checkin_conv = ConversationHandler(
        entry_points=[CommandHandler("run_evening", cmd_run_evening)],
        states={
            CHECKIN_MOOD: [MessageHandler(filters.TEXT & ~filters.COMMAND, checkin_mood_step)],
            CHECKIN_WORKLOAD: [MessageHandler(filters.TEXT & ~filters.COMMAND, checkin_workload_step)],
            CHECKIN_PRODUCTIVITY: [MessageHandler(filters.TEXT & ~filters.COMMAND, checkin_productivity_step)],
            CHECKIN_NOTE_CHOICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, checkin_note_choice_step)],
            CHECKIN_NOTE_TEXT: [MessageHandler(filters.TEXT & ~filters.COMMAND, checkin_note_text_step)],
            CHECKIN_PLAN_CHOICE: [MessageHandler(filters.TEXT & ~filters.COMMAND, checkin_plan_choice_step)],
            CHECKIN_PLAN_TIME: [MessageHandler(filters.TEXT & ~filters.COMMAND, checkin_plan_time_step)],
        },
        fallbacks=[],
    )

    # Сначала конверсейшн
    app.add_handler(checkin_conv)

    # Инлайн-фидбек
    app.add_handler(CallbackQueryHandler(advice_feedback_callback, pattern=r"^advfb:"))
    # Инлайн-оценка осознанности 1–5
    app.add_handler(CallbackQueryHandler(awareness_callback, pattern=r"^aware:"))

    # Текст для "Другое…" — ОБЯЗАТЕЛЬНО block=False
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            advice_custom_rest_note_step,
            block=False,
        )
    )

    # Ответ на вопрос про осознанность (1–5), тоже block=False
    app.add_handler(CommandHandler("run_morning", cmd_run_morning))
    app.add_handler(CommandHandler("run_day", cmd_run_day))

    # Сервисная команда для ручного запуска отчёта
    app.add_handler(CommandHandler("run_report", cmd_run_report))

    return app


if __name__ == "__main__":
    application = build_application()
    application.run_polling()
