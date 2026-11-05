# blender_stroika

Процедурный коралл в Blender **без GUI**: базовая сетка и дерево Geometry Nodes
строятся целиком из Python, морфология — по референсу
`reference/Qwen21_00002.png` (пухлое основание с сотами, скрученная талия,
два веера с радиальными складками и ровными дырочками у кромок).

![coral](renders/coral_final.png)

## Состав

| Путь | Что это |
| --- | --- |
| `scripts/coral_gn.py` | генератор: тело вращения (61.7k вершин) + модификатор CoralGN (228 нод) |
| `reference/Qwen21_00002.png` | целевая морфология |
| `renders/coral_final.png` | финальный рендер (Cycles 1000×1100, 72 spp) |
| `renders/draft_v10_d.png → draft_v10_h.png` | прогресс итераций |
| `renders/draft_v10_54.png` | контрольный рендер на Blender 5.4.0 Alpha |
| `coral.blend` | готовая сцена (слайдеры в модификаторе CoralGN) |
| `worklog.md` | журнал работ Task 1–5 |

## Запуск

```bash
blender -b --factory-startup -P scripts/coral_gn.py            # просто собрать
CORAL_LOW=1  CORAL_OUT=draft.png  blender -b --factory-startup -P scripts/coral_gn.py   # черновик ~10 c
CORAL_BLEND=coral.blend CORAL_OUT=final.png blender -b --factory-startup -P scripts/coral_gn.py  # финал + blend
```

Переменные: `CORAL_LOW` (черновое качество), `CORAL_OUT` (png), `CORAL_BLEND`
(сохранение сцены), `CORAL_DIAG` (bbox/счётчики), `CORAL_STAGE` (бисект цепочки:
`twist|wobble|pleats|lobes|holes|blur|bool`).

## Слайдеры модификатора CoralGN

| Слайдер | Дефолт | За что отвечает |
| --- | --- | --- |
| `Twist` | 0.70 | скрутка талии |
| `Wobble` | 0.012 | лёгкая органическая неровность |
| `Fan Spread` | 0.62 | разлёт вееров в стороны |
| `Fan Curl` | 0.30 | загиб кромок вееров вниз |
| `Gap Rise` | 0.25 | подъём зоны между веерами (V-вырез) |
| `Lobe Sharp` | 4.2 | раздельность двух вееров |
| `Pleat Count` | 36 | число радиальных складок |
| `Pleat Depth` | 0.040 | глубина складок |
| `Holes Enable` | 1.0 | вкл/выкл ямки |
| `Hole Spacing` | 0.105 | шаг сотовой решётки |
| `Hole Radius` | 0.062 | радиус ямок основания |
| `Hole Depth` | 0.050 | глубина ямок |
| `Hole Var` | 0.70 | разброс размера ямок |
| `Top Hole Radius` | 0.024 | мелкие дырочки на веерах |
| `Smooth` | 4 | итерации финального сглаживания |
| `Cutter Scale` | 1.0 | масштаб объекта-резака |
| `Cutter Object` | — | объект для Mesh Boolean (Difference) |
| `Extra Object` | — | своя геометрия, деформируется вместе с кораллом |

Любой слайдер можно задать через env: `GN_FAN_SPREAD=0.8 GN_HOLE_DEPTH=0.06 ...`.

## Проверено на версиях

- Blender **5.3.0 Beta** (`f75ffda59776`) — основная.
- Blender **5.4.0 Alpha** (`6f4fc760c138`) — тот же скрипт без правок, результат
  побайтово совпадает (bbox ±1.165 / h 1.800, 61 697 вершин). Версия вторична —
  решает параметризация.
