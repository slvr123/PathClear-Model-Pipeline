"""Build Sean's two-page PathClear collection cheat sheet."""
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'output/pdf/PathClear_Dataset_Collection_Cheat_Sheet.pdf'
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
W, H = A4
M = 40
INK = colors.HexColor('#111111')
GRAY = colors.HexColor('#555555')
styles = {
    'body': ParagraphStyle('body', fontName='Helvetica', fontSize=10, leading=13, textColor=INK),
    'small': ParagraphStyle('small', fontName='Helvetica', fontSize=9, leading=11.5, textColor=INK),
    'table': ParagraphStyle('table', fontName='Helvetica', fontSize=9.3, leading=11.8, textColor=INK),
    'title': ParagraphStyle('title', fontName='Helvetica-Bold', fontSize=12, leading=14.5, textColor=INK),
}
c = canvas.Canvas(str(OUTPUT), pagesize=A4)
c.setTitle('PathClear Dataset Collection Cheat Sheet')
c.setAuthor('Sean')
c.setSubject('Local collection examples, ten classes, and pilot annotation checks')


def para(text, x, top, width, style='body'):
    p = Paragraph(text, styles[style])
    _, height = p.wrap(width, H)
    p.drawOn(c, x, top - height)
    return top - height


def header(kicker, title, subtitle):
    c.setFillColor(GRAY)
    c.setFont('Helvetica-Bold', 9)
    c.drawString(M, H - 38, kicker)
    c.setFillColor(INK)
    c.setFont('Helvetica-Bold', 23)
    c.drawString(M, H - 68, title)
    para(subtitle, M, H - 82, W - 2 * M, 'small')
    c.setStrokeColor(INK)
    c.setLineWidth(1)
    c.line(M, H - 116, W - M, H - 116)


def footer(page):
    c.setFillColor(GRAY)
    c.setFont('Helvetica', 8)
    c.drawString(M, 26, 'Sean | PathClear | 30 September 2026 | Pilot rules to review on real images')
    c.drawRightString(W - M, 26, f'{page} / 2')


header('FIELD SHEET / 01', 'Collect useful local footage',
       'My starting task: collect varied walking scenes, then review 20-30 annotated pilot frames.<br/>'
       'That pilot checks the workflow; it is not a sufficient training-size target.')

cards = [
    ('1. Match the camera viewpoint',
     'If my final camera sits on glasses, I record from near eye level, facing the walking direction. '
     'A temporary phone goes near that same position, not at waist height or pointed straight down.<br/><br/>'
     '<b>I check:</b> height, tilt, orientation and field of view. I record a short walk and confirm that nearby pavement and the path ahead are visible. I note the actual setup; no universal angle is fixed yet.'),
    ('2. Keep the whole walking scene',
     'I include near, middle-distance and farther objects, side views and partial occlusion.<br/><br/>'
     '<b>Example:</b> I record an approach to stairs, with the surrounding pavement and people still visible, rather than collect only stair close-ups.<br/><br/>'
     '<b>I check:</b> can I recognize object edges at the captured resolution? I retain context and avoid zooming in for every sample.'),
    ('3. Cover all ten classes',
     'I collect Road, Sidewalk, Person, Vehicle, Boundary, Obstacle, Vegetation, Stairs, Crosswalk and Uneven Pavement. Definitions are on page 2.<br/><br/>'
     '<b>I check:</b> do I have multiple examples of each class across different places? I deliberately seek the last three, which the current model has not learned. Every image need not contain all ten.'),
    ('4. Collect look-alike examples',
     'I include intact pavement beside damage, lane lines beside crossing paint, and shadows, paving joints, stains and wet patches.<br/><br/>'
     '<b>Example:</b> I collect both a real pothole and a dark shadow on intact pavement.<br/><br/>'
     '<b>I check:</b> can I see actual damage? Darkness or water alone does not make a region Uneven Pavement.'),
    ('5. Vary places and conditions',
     'I include narrow and wide paths, quiet and busy streets, parked and moving vehicles, sunlight, shade and the lighting I intend the system to handle.<br/><br/>'
     '<b>Example:</b> I collect different crossings on different routes, not many views of one crossing.<br/><br/>'
     '<b>I check:</b> are the important classes present in more than one setting? I note missing conditions instead of assuming coverage.'),
    ('6. Keep originals and notes',
     'I retain full clips, original resolution and unedited source images. I record route/session IDs, date and time, lighting, camera, height, tilt and orientation.<br/><br/>'
     '<b>Example record:</b> R02 / S01 / clip03 / 00:12; phone at glasses height; overcast.<br/><br/>'
     '<b>I check:</b> every selected frame can be traced to its clip and timestamp. I record resolution and frame rate when available.'),
    ('7. Separate locations, not just files',
     'I select meaningfully different frames. I keep neighboring frames, return visits and repeat views of the same location in one split.<br/><br/>'
     '<b>Example:</b> routes A/B train the model; separate route C validates it; separate route D is reserved for final testing. These letters are illustrative, not a required route count.<br/><br/>'
     '<b>I check:</b> no street scene appears across splits. I do not tune from final-test predictions.'),
    ('8. Review a small annotation pilot',
     'I start with 20-30 diverse frames and annotate every identifiable pixel covered by the ten classes. I include normal scenes and examples of stairs, crossing paint and pavement damage.<br/><br/>'
     '<b>I check:</b> a second reviewer inspects some of the same masks. We resolve disagreements before expanding. I count class coverage across frames and routes; many adjacent frames are not independent examples.'),
]
gap = 22
cw = (W - 2 * M - gap) / 2
top = H - 133
row_height = 146
for index, (title, body) in enumerate(cards):
    row, col = divmod(index, 2)
    x, y = M + col * (cw + gap), top - row * row_height
    bottom = para(title, x, y, cw, 'title')
    bottom = para(body, x, bottom - 7, cw, 'small')
    assert bottom >= y - row_height + 8, (title, bottom, y - row_height + 8)
    if row < 3:
        c.setStrokeColor(colors.HexColor('#cccccc'))
        c.setLineWidth(.4)
        c.line(x, y - row_height + 7, x + cw, y - row_height + 7)
para('<b>Before leaving:</b> I play back a clip, check focus/exposure and confirm that my route and camera notes are saved.',
     M, 105, W - 2 * M, 'small')
footer(1)
c.showPage()

header('LABEL SHEET / 02', 'Ten classes and review checks',
       'I label visible pixels, not hidden surfaces, movement, distance or estimated danger.<br/>'
       'One pixel gets one class. A person covering a crosswalk remains Person.')

rows = [
    ('0', 'Road', 'Vehicle roadway, including ordinary lane lines, arrows and stop lines.'),
    ('1', 'Sidewalk', 'Pedestrian pavement, paved footpaths, curb ramps and visible curb edges. An unmarked road shoulder is not automatically Sidewalk.'),
    ('2', 'Person', 'People, including riders. I label the visible motorcycle or bicycle separately as Vehicle.'),
    ('3', 'Vehicle', 'Cars, jeepneys, buses, trucks, trains, motorcycles, bicycles and motorized or pedal tricycles. Parked vehicles use the same class.'),
    ('4', 'Boundary', 'Walls, fences, gates and railings that form barriers. Building facades stay Ignore.'),
    ('5', 'Obstacle', 'Poles, signs/lights, bollards, cones, bins, boxes, loose debris and distinct freestanding stall structures. A more specific class takes priority.'),
    ('6', 'Vegetation', 'Trees, trunks, shrubs, grass and plants, even when they extend into the path.'),
    ('7', 'Stairs', 'Visible stair treads and risers, including an identifiable constructed step. Flat landings use their surface class; handrails are Boundary.'),
    ('8', 'Crosswalk', 'Visible paint belonging to a pedestrian crossing. Unpainted gaps remain Road. I do not fill the whole crossing zone.'),
    ('9', 'Uneven Pavement', 'Clearly broken, sunken, raised, missing or displaced paved surface, including visible pothole interiors. No guesses from shadows or texture.'),
]
data = [[Paragraph('<b>' + value + '</b>', styles['table']) for value in ('ID', 'Class', 'What I label / example')]]
data += [[Paragraph(escape(value), styles['table']) for value in row] for row in rows]
t = Table(data, colWidths=[25, 105, W - 2 * M - 130])
t.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#eeeeee')),
    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
    ('LINEBELOW', (0, 0), (-1, 0), .8, INK),
    ('LINEBELOW', (0, 1), (-1, -1), .3, colors.HexColor('#cccccc')),
    ('LEFTPADDING', (0, 0), (-1, -1), 5), ('RIGHTPADDING', (0, 0), (-1, -1), 5),
    ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
]))
_, th = t.wrap(W - 2 * M, H)
table_top = H - 132
t.drawOn(c, M, table_top - th)
y = table_top - th - 12
y = para('<b>255 = Ignore, not an eleventh class.</b> I use it for sky, building facades, unpaved terrain, '
         'unsupported content and uncertain pixels. Open drains and animals are coverage gaps under the current rules; '
         'I record them rather than force them into a class.', M, y, W - 2 * M, 'small')
y -= 17
y = para('My review before expanding the dataset', M, y, W - 2 * M, 'title')
y -= 7
checks = [
    '<b>Agree on boundaries:</b> crossing paint only; visible pavement damage only; ordinary curb = Sidewalk. '
    'I review ambiguous cases together and keep unresolved pixels at 255.',
    '<b>Inspect overlays:</b> masks align with the original image, thin objects are included, and labels follow visible edges. '
    'I do not use bounding rectangles as segmentation masks.',
    '<b>Check exports:</b> masks are original-size, single-channel 8-bit PNGs containing IDs 0-9 or 255. '
    'A colored preview or JPEG is not the training mask.',
    '<b>Check coverage and separation:</b> each split has useful examples of the classes being evaluated, '
    'with no repeated locations across splits. I expand collection where examples are missing.',
]
for item in checks:
    y = para(item, M, y, W - 2 * M, 'small') - 7
assert y > 68, ('page 2 overflow', y)
para('Project references: docs/PATHCLEAR_ANNOTATION_GUIDE.md and docs/LOCAL_DATASET_PLAN.md.<br/>'
     'I use these draft rules for the pilot and settle disagreements before large-scale annotation.',
     M, 60, W - 2 * M, 'small')
footer(2)
c.save()

reader = PdfReader(OUTPUT)
assert len(reader.pages) == 2
text = '\n'.join(page.extract_text() for page in reader.pages)
for term in ['Road', 'Sidewalk', 'Person', 'Vehicle', 'Boundary', 'Obstacle', 'Vegetation', 'Stairs', 'Crosswalk', 'Uneven Pavement', '20-30', '255']:
    assert term in text, term
print(f'Created {OUTPUT}; 2 pages; all ten classes and pilot checks present.')
