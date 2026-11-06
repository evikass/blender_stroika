# blender_neuro — симулятор самообучающихся существ

Домашняя лаборатория эволюции поведения: в окне программы живёт **трёхмерный
мир с гравитацией и физикой предметов**, в нём находится существо на костной
системе (руки, ноги, голова, туловище — всё на суставах), и оно **само
обучается** взаимодействовать с этим миром методом проб и ошибок
(обучение с подкреплением, PPO).

Результаты обучения (походки, движения) **экспортируются в Blender** —
двумя способами: BVH-мокап и родной аддон, который строит арматуру, меши
и декорации сам.

Заточено под мультипроекты: кроме гуманоида есть шаблон четвероногих —
лиса, волк, медведь, заяц, барсук, кот и ёжик. Учите героев ходить —
и переносите их движения в свои сцены.

## Состав существ

| Пресет | Кто это | Суставов | Размер |
|---|---|---|---|
| `human` | Детализированный гуманоид ~1.7 м | 17 | 1.0 |
| `fox` | Лиса | 13 | 1.0 |
| `wolf` | Волк | 13 | 1.15 |
| `bear` | Медведь | 13 | 1.55 |
| `hare` | Заяц | 13 | 0.60 |
| `badger` | Барсук | 13 | 0.85 |
| `cat` | Кот | 13 | 0.75 |
| `hedgehog` | Ёж (детектив 🙂) | 13 | 0.55 |

## Задачи обучения

1. **walk** — ходьба вперёд по полосе препятствий (ящики, рампа).
2. **balance** — стоять и удерживать равновесие под случайными толчками.
3. **target** — дойти до жёлтого маркера в лесном мире с домиком и деревьями.

## Установка

Нужен Python 3.10–3.12 (64-бит).

```bash
cd blender_neuro
python -m venv venv
# Windows:  venv\Scripts\activate
# Linux/Mac: source venv/bin/activate
pip install -r requirements.txt

# PyTorch отдельно:
# RTX 5060 Ti (Blackwell, CUDA 12.8):
pip install torch --index-url https://download.pytorch.org/whl/cu128
# если нет NVIDIA-карты:
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Запуск окна-симулятора

```bash
python main.py                                  # гуманоид, задача "ходьба"
python main.py --task target --creature fox     # лиса идёт к цели в лесу
python main.py --task balance --window 1000x700
python main.py --task walk --load outputs/ckpt_walk_human.zip --watch
```

Окно открывается 800x600 и **разворачивается** на весь экран обычной
кнопкой или двойным кликом по заголовку. (Заголовок окна служебный —
так устроен рендерер PyBullet; на содержимое не влияет.)

### Управление (кнопки в окне + горячие клавиши)

| Клавиша | Действие |
|---|---|
| `1` / `2` / `3` | задача: ходьба / баланс / цель |
| `W` | **Смотреть** — прогон текущей политики в окне |
| `T` | **Турбо-обучение** фоном (без графики) на N млн шагов |
| `S` | стоп турбо |
| `R` | сброс существа |
| `K` | сохранить чекпоинт |
| `E` | **экспорт** BVH + JSON в `outputs/` |
| `C` | камера-слежение вкл/выкл (в свободной камере работайте мышью) |

Слайдеры в окне: номер задачи, скорость просмотра (шагов на кадр),
размер турбо-порции (1/2/5/10/20/50 млн шагов).

### Как это выглядит по времени (ориентиры)

| Железо | Скорость турбо | Ходьба «расхристанного рэгдолла» | Устойчивая походка |
|---|---|---|---|
| RTX 5060 Ti + 6 сред | ~4–8 тыс. шагов/с | 1–3 млн шагов (~10 мин) | 10–30 млн (~1–2 ч) |
| Только CPU (6 ядер) | ~2–3 тыс. шагов/с | 1–3 млн (~20 мин) | 10–30 млн (~3–5 ч) |

Физика считается на CPU, нейросеть — на GPU; для RTX 5060 Ti ставьте
torch с CUDA 12.8 (см. выше).

## Турбо-обучение без окна (максимально быстро)

```bash
python main.py --headless --task walk --steps 10000000 --nenvs 6
# затем посмотреть результат:
python main.py --task walk --load outputs/ckpt_walk_human.zip --watch
```

Чекпоинты автосохраняются в `outputs/ckpt_<задача>_<существо>.zip` —
турбо-кнопка в окне продолжает с них обучение между сеансами.

## Экспорт в Blender

После нажатия `E` в `outputs/` появляются два файла:

- `..._<штамп>.bvh` — мокап движения;
- `..._<штамп>.json` — то же + спека существа + декорации мира.

### Способ 1. BVH (быстрый просмотр)

1. Blender → `File → Import → Motion Capture (.bvh)`.
2. Scale = `1.0` (файлы в метрах).
3. Арматура с анимацией готова; нажмите Space — движение проиграется.

### Способ 2. Аддон (полная сборка сцены)

1. Blender → `Edit → Preferences → Add-ons → Install…` → выбрать
   `blender_neuro_importer.py` → включить галочку.
2. `File → Import → Neuro Sim Motion (.json)` → выбрать JSON из `outputs/`.
3. Аддон сам построит:
   - арматуру с теми же костями и суставами, что в физике;
   - капсульные меши частей тела в цветах симулятора (галочка «Меши существа»);
   - декорации мира: деревья, камни, домик, ящики (галочка «Мир из файла»);
   - анимацию по всем кадрам.
4. `N-панель → Neuro` — кнопки импорта всегда под рукой.

Оба способа дают **физически точное** движение: кинематика аддона
сверена с симулятором с точностью до микрона (см. `tests/test_headless.py`).

## Свои существа и миры

Существо — это JSON-спека (`presets/human.json`):

- `links[]` — звенья: `parent`, сустав `anchor/axis/lo/hi/max_force`,
  `com`, `mass`, фигура `capsule/box/sphere` с цветом;
- координаты — в мировых осях покоя (Z вверх), ступни на земле.

Скрипт-шаблон четвероногого: `python -c "from neurosim.spec import *"`
→ `make_quadruped_spec(name="dog", scale=0.9, color=(...))` и `save()`.
Любой JSON-мир (`presets/world_*.json` по аналогии) подключается в
`neurosim/env.py` → словарь `WORLDS`.

## Структура

```
blender_neuro/
├── main.py                    # точка входа (окно + headless)
├── blender_neuro_importer.py  # аддон Blender
├── neurosim/
│   ├── spec.py     # генератор существ (гуманоид + звери)
│   ├── build.py    # спека -> URDF -> PyBullet
│   ├── world.py    # миры: арена / полоса препятствий / лес с домиком
│   ├── env.py      # Gymnasium-среда: 3 задачи, награды, наблюдения
│   ├── train.py    # PPO (Stable-Baselines3), турбо-режим
│   ├── export.py   # BVH + motion JSON
│   ├── fk.py       # прямая кинематика (общая с аддоном)
│   └── gui.py      # окно 800x600: кнопки, HUD, камера-слежение
├── presets/        # human.json, fox.json ... + миры
├── tests/test_headless.py     # самопроверка установки
└── outputs/        # чекпоинты и экспорт (создаётся автоматически)
```

## Проверка установки

```bash
python tests/test_headless.py
```

Проверяет: гуманоид собирается и стоит, кинематика совпадает с физикой,
PPO делает шаг обучения, BVH/JSON экспортируются.

## Известные ограничения

- Столкновения частей тела друг с другом отключены (руки проходят сквозь
  туловище) — так симуляция быстрее и стабильнее; с миром всё сталкивается.
- Односуставные шарниры (нет скрутки предплечья и пр.).
- Обучение зверей не отлажено так тщательно, как гуманоида — шаблон тот же,
  но, возможно, потребуются другие порции шагов.
- Заголовок окна служебный (PyBullet); размер окна меняется свободно.

---

# blender_neuro (English)

A home lab for evolving behavior: a **3D world with gravity and object
physics**, a **bone-rigged creature** (arms, legs, head, torso — all on
joints) that **teaches itself** to act through trial and error
(reinforcement learning, PPO). Learned motions **export to Blender** as
BVH motion capture or via a native add-on that builds the armature, meshes
and scenery for you.

Built for animation projects: a detailed humanoid plus a quadruped
template — fox, wolf, bear, hare, badger, cat, hedgehog.

**Tasks:** walk forward across obstacles · keep balance under random
pushes · reach a glowing marker in a forest world.

**Install** (Python 3.10–3.12):

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu128  # RTX 50xx
# or: pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**Run:** `python main.py` — an 800x600 resizable window with on-screen
buttons and hotkeys: `1/2/3` task, `W` watch current policy, `T` background
turbo-training (1–50M steps), `S` stop, `R` reset, `K` save, `E` export,
`C` camera follow.

**Fast training:** `python main.py --headless --task walk --steps 10000000 --nenvs 6`,
then `python main.py --task walk --load outputs/ckpt_walk_human.zip --watch`.
On an RTX 5060 Ti expect a rough gait after 1–3M steps and a steady walk
after 10–30M.

**Blender export:** press `E` → files appear in `outputs/`:

- import `.bvh` via *File → Import → Motion Capture* (Scale = 1.0), or
- install `blender_neuro_importer.py` as an add-on and use
  *File → Import → Neuro Sim Motion (.json)* — it rebuilds the armature,
  body meshes, world props and the animation exactly as simulated.

Creature and world presets are plain JSON (`presets/`), so you can design
new creatures — the simulator, the BVH exporter and the Blender add-on all
read the same format. Self-check: `python tests/test_headless.py`.
