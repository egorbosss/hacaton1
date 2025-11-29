# ДЛЯ РАБОТЫ С JSON
'''
import pygsheets
import json
import numpy as np  # можно не использовать, но пусть остаётся

# ---------- чтение json ----------
JSON_PATH = 'tracking_results.json'
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

# ---------- авторизация google sheets ----------
gc = pygsheets.authorize(client_secret='keys/client_secret.json')

# ┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┓
SPREADSHEET_NAME = 'Tracking Data'  #  ┇ ИМЯ ТАБЛИЦЫ
# ┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┅┛

try:
    sh = gc.open(SPREADSHEET_NAME)   # пробуем открыть
except pygsheets.SpreadsheetNotFound:
    sh = gc.create(SPREADSHEET_NAME)  # если нет - создаём

# можно оставить sheet1 как есть или не трогать
wks = sh.sheet1

# доступ всем по ссылке (только чтение)
sh.share(None, role='reader', type='anyone')

# ---------- лист Summary ----------
try:
    ws_summary = sh.worksheet_by_title('Summary')
except pygsheets.WorksheetNotFound:
    ws_summary = sh.add_worksheet('Summary', rows=100, cols=10)

ws_summary.clear()

summary_header = ['id', 'last_action', 'last_frame', 'last_x', 'last_y']
ws_summary.update_row(1, summary_header)

summary_rows = []
for person in data:
    pid = person.get('id')
    last_action = person.get('last_action')
    last_frame = person.get('last_frame')
    last_pos = person.get('last_position', {})
    last_x = float(last_pos.get('x', 0.0))
    last_y = float(last_pos.get('y', 0.0))
    summary_rows.append([pid, last_action, last_frame, last_x, last_y])

if summary_rows:
    ws_summary.update_values('A2', summary_rows)

# ---------- лист Trajectories ----------
try:
    ws_traj = sh.worksheet_by_title('Trajectories')
except pygsheets.WorksheetNotFound:
    ws_traj = sh.add_worksheet('Trajectories', rows=5000, cols=10)

ws_traj.clear()

traj_header = ['id', 'step', 'x', 'y']
ws_traj.update_row(1, traj_header)

traj_rows = []
for person in data:
    pid = person.get('id')
    traj = person.get('trajectory', [])
    for idx, point in enumerate(traj, start=1):
        x = float(point.get('x', 0.0))
        y = float(point.get('y', 0.0))
        traj_rows.append([pid, idx, x, y])

if traj_rows:
    ws_traj.update_values('A2', traj_rows)

    # ====== СОЗДАНИЕ ДИАГРАММЫ ПО ТРАЕКТОРИЯМ ======
    # последняя строка с данными (1 — шапка)
    last_row = len(traj_rows) + 1

    # домен (ось X диаграммы) — шаг (step)
    domain = (f'B2', f'B{last_row}')

    # значения (ряды) — координаты x и y
    ranges = [
        (f'C2', f'C{last_row}'),  # X
        (f'D2', f'D{last_row}')   # Y
    ]

    # создаём диаграмму (тип по умолчанию, потом можно руками переключить на Scatter)
    ws_traj.add_chart(domain, ranges, 'Trajectories chart')

print('Готово! Данные записаны в Google Sheets:', SPREADSHEET_NAME)
print('Листы: "Summary" и "Trajectories", на "Trajectories" добавлена диаграмма.')
'''