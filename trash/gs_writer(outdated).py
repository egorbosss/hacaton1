#google sheets writer
'''
# gs_writer.py ДЛЯ РАБОТЫ ОНЛАЙН
import pygsheets

# Авторизация
gc = pygsheets.authorize(client_secret='keys/client_secret.json')

SPREADSHEET_NAME = 'Tracking Realtime'
SHEET_TITLE = 'States'   # лист с текущими состояниями людей

# Открываем / создаём таблицу
try:
    sh = gc.open(SPREADSHEET_NAME)
except pygsheets.SpreadsheetNotFound:
    sh = gc.create(SPREADSHEET_NAME)

# Открываем / создаём лист
try:
    ws = sh.worksheet_by_title(SHEET_TITLE)
except pygsheets.WorksheetNotFound:
    ws = sh.add_worksheet(SHEET_TITLE, rows=100, cols=10)

# Чистим лист и ставим шапку ОДИН РАЗ при импорте модуля
ws.clear()
ws.update_row(1, ['person_id', 'frame', 'action', 'x', 'y'])

# Здесь будем хранить: person_id -> номер строки
_person_row_map = {}


def _get_row_for_person(person_id: int) -> int:
    """Находит строку для person_id или создаёт новую."""
    global _person_row_map

    if person_id in _person_row_map:
        return _person_row_map[person_id]

    # Читаем колонку A (person_id), ищем, нет ли уже такой записи
    col = ws.get_col(1, include_tailing_empty=False)
    # col[0] - это заголовок 'person_id'
    for idx, val in enumerate(col[1:], start=2):  # начинаем со 2-й строки
        try:
            if int(val) == int(person_id):
                _person_row_map[person_id] = idx
                return idx
        except (TypeError, ValueError):
            continue

    # Если не нашли - создаём новую строку в конце
    new_row = len(col) + 1  # следующая свободная строка
    ws.update_value(f"A{new_row}", int(person_id))  # записали id в A
    _person_row_map[person_id] = new_row
    return new_row


def append_event(frame_idx: int, person_id: int, action: str, x: float, y: float):
    """
    Обновляет состояние человека:
    всегда пишет в одну и ту же строку для данного person_id.
    """
    row = _get_row_for_person(person_id)

    # Формируем строку: [id, frame, action, x, y]
    values = [
        int(person_id),
        int(frame_idx),
        str(action),
        float(x),
        float(y),
    ]

    # Обновляем всю строку начиная с A
    ws.update_row(row, values)
'''