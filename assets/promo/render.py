"""Build the X graphic from values and plotted points published in arXiv:2609.09835."""
from pathlib import Path
from html import escape

HERE = Path(__file__).resolve().parent
INK = '#17243a'
BLUE = '#3169b8'
PURPLE = '#9c79d5'
MUTED = '#5d6c80'


def text(x, y, value, size=28, weight=400, color=INK, extra=''):
    return f'<text x="{x}" y="{y}" font-family="DejaVu Sans,Arial,sans-serif" font-size="{size}" font-weight="{weight}" fill="{color}" {extra}>{escape(value)}</text>'


def rect(x, y, w, h, fill, r=26, stroke='#263956', sw=3):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


parts = ['''<svg xmlns="http://www.w3.org/2000/svg" width="2400" height="1700" viewBox="0 0 2400 1700">
<rect width="2400" height="1700" fill="#f8f7f2"/>
<rect width="2400" height="425" fill="#13233d"/>
<circle cx="2230" cy="-80" r="340" fill="#244c8a" opacity=".48"/>
<circle cx="2060" cy="275" r="148" fill="#a9d9cd" opacity=".18"/>
<path d="M1815 243 C1900 140 1990 190 2060 115 C2140 30 2260 75 2350 35" fill="none" stroke="#93d4ad" stroke-width="10" opacity=".55"/>
<circle cx="1815" cy="243" r="15" fill="#93d4ad"/><circle cx="2060" cy="115" r="15" fill="#93d4ad"/><circle cx="2350" cy="35" r="15" fill="#93d4ad"/>
''']
parts += [
    text(78, 80, 'HYPERTRACE', 28, 900, '#99d9ae', 'letter-spacing="5"'),
    rect(1950, 50, 370, 60, '#203c66', 30, '#5e88b2', 2),
    text(2135, 91, 'FINDINGS · EMNLP 2026', 22, 800, '#f8f7f2', 'text-anchor="middle"'),
    text(75, 190, 'When the topic changes,', 76, 900, '#ffffff'),
    text(75, 280, 'does your AI still know you?', 76, 900, '#ffffff'),
    text(80, 354, 'Preference tracing that adapts now and remembers what lasts.', 33, 500, '#c9d7e8'),
    rect(66, 460, 1460, 1070, '#ffffff', 28, '#233450', 4),
    rect(1556, 460, 777, 515, '#f0f2ff', 28, '#233450', 4),
    rect(1556, 1000, 777, 530, '#e8f6ec', 28, '#233450', 4),
    rect(90, 495, 270, 52, '#ddecff', 26, 'none', 0),
    text(120, 531, '01 / ADAPTATION', 24, 800, '#2f61aa'),
    text(105, 604, 'Tracks you across sessions', 45, 800),
    text(105, 647, 'PRISM · preference prediction accuracy', 27, 500, MUTED),
    rect(106, 685, 160, 50, '#e4efff', 25, 'none', 0),
    text(186, 720, 'Ours', 27, 800, BLUE, 'text-anchor="middle"'),
    rect(286, 685, 245, 50, '#f1e9ff', 25, 'none', 0),
    text(408, 720, 'Memory-based', 27, 800, PURPLE, 'text-anchor="middle"'),
    '<line x1="1198" y1="706" x2="1198" y2="1418" stroke="#dfe5ed" stroke-width="3"/>',
    text(1230, 805, '61.4%', 67, 900, BLUE),
    text(1230, 849, 'Ours', 29, 800),
    text(1230, 887, 'after turn 20', 23, 500, MUTED),
    '<line x1="1230" y1="925" x2="1460" y2="925" stroke="#d8deea" stroke-width="3"/>',
    text(1230, 992, '54.4%', 53, 800, PURPLE),
    text(1230, 1032, 'Memory-based', 24, 700),
    rect(1218, 1115, 267, 190, '#e8f5eb', 20, 'none', 0),
    text(1240, 1170, '41 / 50', 35, 900, '#308457'),
    text(1240, 1212, 'users start a new', 23, 700),
    text(1240, 1251, 'session near', 23, 500),
    text(1240, 1290, 'turn 20', 23, 500),
    rect(1584, 495, 325, 52, '#dee6ff', 26, 'none', 0),
    text(1614, 531, '02 / LESS FEEDBACK', 23, 800, '#3e61a9'),
    text(1618, 607, '80% feedback withheld', 38, 800),
    text(1618, 650, 'Can it still learn your preferences?', 24, 500, MUTED),
    rect(1584, 1035, 337, 52, '#d5efd9', 26, 'none', 0),
    text(1614, 1071, '03 / TOPIC SHIFTS', 23, 800, '#2d8050'),
    text(1618, 1147, 'Frequent session changes', 37, 800),
    text(1618, 1191, 'Does it stay with you?', 24, 500, MUTED),
]

# The six y coordinates for each series come from the paper's original Figure 3 SVG.
# Its plot maps raw y=337.4201 to accuracy .3 and raw y=813.1279 to .8.
turns = [0, 5, 10, 15, 20, 25]
raw = {
    'ours': [425.76585, 527.70327, 500.51994, 575.274, 580.3031, 722.9985],
    'memory': [446.15333, 493.7241, 493.7241, 534.4991, 434.3284, 729.65176],
}
values = {k: [.3 + (y - 337.4201) * .5 / (813.1279 - 337.4201) for y in ys] for k, ys in raw.items()}
px0, px1, py0, py1 = 170, 1140, 1320, 776
X = lambda turn: px0 + (turn / 25) * (px1 - px0)
Y = lambda value: py0 - (value - .3) / .5 * (py0 - py1)
parts += [rect(130, 760, 1035, 655, '#fbfdff', 20, '#e3e9f0', 2)]
for tick in [.3, .4, .5, .6, .7, .8]:
    yy = Y(tick)
    parts += [f'<line x1="{px0}" y1="{yy:.1f}" x2="{px1}" y2="{yy:.1f}" stroke="#e8edf2" stroke-width="2"/>',
              text(153, yy+9, f'{tick:.1f}', 23, 500, MUTED, 'text-anchor="end"')]
for turn in turns:
    xx = X(turn)
    parts += [f'<line x1="{xx:.1f}" y1="{py0}" x2="{xx:.1f}" y2="{py1}" stroke="#eef1f5" stroke-width="2"/>',
              text(xx, 1370, str(turn), 23, 500, MUTED, 'text-anchor="middle"')]
parts += [
    f'<rect x="{X(18):.1f}" y="{py1}" width="{X(22)-X(18):.1f}" height="{py0-py1}" fill="#dff1e5" opacity=".72"/>',
    f'<line x1="{X(20):.1f}" y1="{py1}" x2="{X(20):.1f}" y2="{py0}" stroke="#5ba977" stroke-width="3" stroke-dasharray="12 10"/>',
    text(X(20)-8, 806, 'session switch', 22, 800, '#388556', 'text-anchor="end"'),
]
for key, color, width in [('memory', PURPLE, 7), ('ours', BLUE, 10)]:
    coords = [(X(turn), Y(value)) for turn, value in zip(turns, values[key])]
    pts = ' '.join(f'{x:.1f},{y:.1f}' for x, y in coords)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linecap="round" stroke-linejoin="round"/>')
    for x, y in coords:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{9 if key=="ours" else 8}" fill="#fff" stroke="{color}" stroke-width="{4 if key=="ours" else 3}"/>')
parts += [
    text(650, 1403, 'Interaction turn', 23, 600, MUTED, 'text-anchor="middle"'),
    text(105, 1484, 'More stable around the main session transition', 27, 700, '#3169b8'),
]


def bars(y, ours, memory, gain):
    x, width = 1620, 590
    for label, value, color, yy in [('Ours', ours, BLUE, y), ('Memory-based', memory, PURPLE, y+115)]:
        parts.extend([text(x, yy-17, label, 29, 800),
                      text(2230, yy-17, f'{value*100:.1f}%', 30, 900, color, 'text-anchor="end"'),
                      rect(x, yy, width, 44, '#e3e8f1', 22, 'none', 0),
                      rect(x, yy, round(value/.65*width), 44, color, 22, 'none', 0)])
    parts.append(text(x, y+220, f'+{gain:.1f} points', 35, 900, BLUE))

bars(716, .5567, .4582, 9.9)
bars(1258, .5304, .4224, 10.8)
parts += ['<line x1="68" y1="1584" x2="2330" y2="1584" stroke="#b9c7cf" stroke-width="2"/>',
          text(75, 1634, 'PRISM  ·  Preference prediction  ·  arXiv:2609.09835', 27, 600, '#43546d'),
          '</svg>']

(HERE / 'hypertrace-results.svg').write_text('\n'.join(parts))
try:
    import cairosvg
except ImportError:
    pass
else:
    cairosvg.svg2png(url=str(HERE / 'hypertrace-results.svg'), write_to=str(HERE / 'hypertrace-results.png'))
