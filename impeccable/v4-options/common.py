ROWS = [
 ("TabM","0.9585 (6)","0.9954 (1)","0.9941 (1)","0.9517 (9)"),
 ("LightGBM","0.9589 (4)","0.9951 (2)","0.9932 (4)","0.9549 (2)"),
 ("CatBoost, all categorical","0.9603 (1)","0.9950 (3)","0.9934 (3)","0.9537 (5)"),
 ("RealMLP","0.9580 (7)","0.9950 (4)","0.9937 (2)","0.9534 (8)"),
 ("XGBoost, 1,024 bins","0.9592 (3)","0.9949 (5)","0.9928 (6)","0.9549 (3)"),
 ("XGBoost","0.9587 (5)","0.9948 (6)","0.9928 (5)","0.9549 (4)"),
 ("XGBoost + exact-value encoding","0.9600 (2)","0.9946 (7)","0.9922 (7)","0.9550 (1)"),
 ("MLP (scikit-learn)","0.9566 (8)","0.9934 (8)","0.9921 (8)","0.9534 (7)"),
 ("TabPFN (10k context)","0.9544 (9)","0.9931 (9)","0.9892 (9)","0.9534 (6)"),
 ("Logistic regression","0.9486 (10)","0.9784 (10)","0.9714 (10)","0.9472 (10)"),
]
BEST = {(0,2),(0,3),(2,1),(6,4)}   # (row, col) best in each setting
ARROW = '<svg class="arr" viewBox="0 0 44 18"><use href="#arr"/></svg>'
DEFS = '<svg width="0" height="0" style="position:absolute"><symbol id="arr" viewBox="0 0 44 18"><path d="M1 9h39M32 2l8 7-8 7" fill="none" stroke="currentColor" stroke-width="2.6" stroke-linecap="square"/></symbol></svg>'
def table(cls="t"):
    h = '<table class="%s"><thead><tr><th>Model</th><th>S→S</th><th>R→R</th><th>S→R</th><th>R→S</th></tr></thead><tbody>' % cls
    for r,row in enumerate(ROWS):
        h += '<tr>'
        for c,cell in enumerate(row):
            if c==0: h += '<td>%s</td>' % cell
            else:
                v, k = cell.split(' ')
                b = ' class="best"' if (r,c) in BEST else ''
                h += '<td%s>%s <i>%s</i></td>' % (b, v, k)
        h += '</tr>'
    return h + '</tbody></table>'
SCRIPT = """<script>
const sp=new URLSearchParams(location.search).get('s');
if(sp) document.body.classList.add('single');
document.querySelectorAll('.wrap').forEach((w,i)=>{ if(sp && +sp!==i+1) w.style.display='none'; });
function fit(){document.querySelectorAll('.wrap').forEach(w=>{const W=w.clientWidth,sc=W/1920;const f=w.firstElementChild;f.style.height=(1080*sc)+'px';f.firstElementChild.style.transform='scale('+sc+')';});}
addEventListener('resize',fit);fit();
</script>"""
BASE = """*{box-sizing:border-box;margin:0;padding:0}
html,body{background:#7b7f88}
body{font-family:var(--body)}
.wrap{width:min(1500px,92vw);margin:28px auto;box-shadow:0 10px 40px #0006}
.single .wrap{width:100vw;margin:0;box-shadow:none}
.frame{position:relative;overflow:hidden;width:100%}
.slide{position:absolute;left:0;top:0;width:1920px;height:1080px;transform-origin:0 0;overflow:hidden}
.arr{width:44px;height:18px;display:inline-block;vertical-align:middle;color:currentColor}
table{border-collapse:collapse;font-variant-numeric:tabular-nums}
"""
def page(title, fonts, css, slides):
    body = ''.join('<div class="wrap"><div class="frame"><section class="slide">%s</section></div></div>' % s for s in slides)
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>%s</title>'
            '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
            '<link href="%s" rel="stylesheet"><style>%s%s</style></head><body>%s%s%s</body></html>') % (title, fonts, BASE, css, DEFS, body, SCRIPT)
