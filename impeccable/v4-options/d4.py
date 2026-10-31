from common import *
FONTS="https://fonts.googleapis.com/css2?family=Barlow:wght@300;400;500;600&family=Barlow+Condensed:wght@500;600;700&display=swap"
CSS="""
:root{--navy:#0a1f55;--cyan:#26a9e0;--pale:#e6f3fb;--ink:#0a1f55;--mute:#4f6286;--head:'Barlow Condensed',sans-serif;--body:'Barlow',sans-serif}
.slide{background:#fff;color:var(--navy);font-family:var(--body)}
.hdr{position:absolute;left:96px;top:56px;font:600 22px var(--body);letter-spacing:.2em;text-transform:uppercase;color:var(--cyan)}
.ftr{position:absolute;left:0;right:0;bottom:0;height:64px;border-top:2px solid #cfe0ee;font:500 20px/64px var(--body);color:var(--mute);padding:0 96px;letter-spacing:.04em}
.ftr b{position:absolute;right:0;top:0;width:200px;height:64px;background:var(--navy);color:#fff;text-align:center;font:600 22px/64px var(--body);letter-spacing:.1em}
/* title */
.s1 .ph{position:absolute;left:0;top:0;width:1920px;height:740px;overflow:hidden}
.s1 .ph img{width:100%;height:100%;object-fit:cover;object-position:50% 52%}
.s1 .ph:after{content:"";position:absolute;inset:0;background:linear-gradient(180deg,#0a1f5588 0%,transparent 40%)}
.s1 .band{position:absolute;left:0;right:0;top:740px;height:340px;background:var(--navy);padding:62px 96px;color:#fff}
.s1 .band:before{content:"";position:absolute;left:0;right:0;top:0;height:8px;background:var(--cyan)}
.s1 h1{font:700 128px/.95 var(--head);letter-spacing:.005em;text-transform:uppercase}
.s1 .sub{font:300 34px/1.35 var(--body);color:#cfe6f5;margin-top:22px;max-width:1300px}
.s1 .meta{position:absolute;right:96px;bottom:50px;text-align:right;font:500 24px/1.5 var(--body);letter-spacing:.06em;color:#fff}
.s1 .meta b{color:var(--cyan);font-weight:600}
/* results */
.s2 h2{position:absolute;left:96px;top:100px;font:700 64px/1 var(--head);text-transform:uppercase;width:1230px}
.s2 .t{position:absolute;left:96px;top:290px;width:1180px;font-size:25px}
.s2 .t th{background:var(--navy);color:#fff;font:600 20px var(--body);letter-spacing:.12em;text-transform:uppercase;text-align:right;padding:14px 18px}
.s2 .t th:first-child,.s2 .t td:first-child{text-align:left}
.s2 .t td{padding:12px 18px;text-align:right;border-bottom:1px solid #d7e6f1}
.s2 .t tr:nth-child(even) td{background:#f1f8fd}
.s2 .t td i{font-style:normal;color:#7e93b4;font-size:19px;margin-left:4px}
.s2 .t td.best{background:var(--cyan)!important;color:#fff;font-weight:700}
.s2 .t td.best i{color:#e5f6ff}
.s2 .pan{position:absolute;left:1360px;top:0;width:560px;height:1016px;background:var(--navy);color:#fff;padding:100px 64px}
.s2 .pan .l{font:600 20px var(--body);letter-spacing:.16em;text-transform:uppercase;color:#8fd2f2}
.s2 .pan .n{font:700 176px/.95 var(--head);color:var(--cyan);margin:8px 0 6px}
.s2 .pan p{font:300 27px/1.35 var(--body);color:#dcecf7}
.s2 .pan hr{border:0;height:2px;background:#2a4585;margin:46px 0 40px}
.s2 .pan .n2{font:700 110px/.95 var(--head);color:#fff;margin:8px 0 6px}
.s2 .note{position:absolute;left:96px;top:960px;font:400 22px var(--body);color:var(--mute)}
/* explainer */
.s3{background:var(--pale)}
.s3 h2{position:absolute;left:96px;top:100px;font:700 70px/1 var(--head);text-transform:uppercase;width:1720px}
.s3 .dsb{position:absolute;top:290px;width:520px;height:250px;background:var(--navy);color:#fff;padding:36px 44px}
.s3 .dsb .l{font:600 20px var(--body);letter-spacing:.16em;text-transform:uppercase;color:#8fd2f2}
.s3 .dsb b{display:block;font:700 92px/1 var(--head);margin-top:8px}
.s3 .dsb span{font:300 26px var(--body);color:#cfe6f5}
.s3 svg.flow{position:absolute;left:0;top:0;width:1920px;height:900px}
.s3 .an{position:absolute;font:500 24px/1.35 var(--body);color:var(--navy)}
.s3 .an b{font:700 40px var(--head);display:block;letter-spacing:.02em}
.s3 .an.hot b{color:var(--cyan)}
.s3 .spec{position:absolute;left:96px;right:96px;top:790px;border-collapse:collapse;background:#fff;width:1728px}
.s3 .spec th{background:var(--navy);color:#fff;font:600 20px var(--body);letter-spacing:.12em;text-transform:uppercase;text-align:left;padding:12px 24px}
.s3 .spec td{padding:16px 24px;font:600 30px var(--head);border-right:1px solid #d7e6f1;letter-spacing:.02em}
.s3 .spec td span{display:block;font:400 22px var(--body);color:var(--mute);letter-spacing:0}
"""
s1='''<div class="s1"><div class="ph"><img src="img/sunrise.jpg" alt="Wing of an airliner against a dusk horizon"></div>
<div class="hdr" style="color:#fff">IS460 · Final project presentation</div>
<div class="band"><h1>Cleared for Satisfaction</h1><div class="sub">Evaluating synthetic data with tabular ML ensembles, transfer testing and explainable AI.</div>
<div class="meta"><b>Group 03</b><br>SMU · 2026</div></div></div>'''
s2='''<div class="s2"><div class="hdr">Results · RQ1 model ladder</div><h2>TabM wins on real passengers. CatBoost wins on the synthetic copy.</h2>
%s<div class="pan"><div class="l">Best on real · R→R AUC</div><div class="n">0.9954</div><p>TabM, trained and tested on real passengers.</p><hr>
<div class="l">Best on synthetic · S→S AUC</div><div class="n2">0.9603</div><p>CatBoost, all columns categorical.</p></div>
<div class="note">AUC of ten models in four settings, rank in brackets. S = synthetic, R = real. Sorted by R→R.</div>
<div class="ftr">IS460 · Group 03 · Cleared for Satisfaction<b>2</b></div></div>''' % table('t')
s3='''<div class="s3"><div class="hdr">Method · Validation</div><h2>How we test whether synthetic findings transfer</h2>
<svg class="flow" viewBox="0 0 1920 900" fill="none" stroke="#0a1f55" stroke-width="2.4">
<path d="M616 372 H1304" stroke="#26a9e0" stroke-width="5"/><path d="M1288 356l18 16-18 16" stroke="#26a9e0" stroke-width="5"/>
<path d="M1304 470 H616"/><path d="M632 454l-18 16 18 16"/>
</svg>
<div class="dsb" style="left:96px"><div class="l">Synthetic survey</div><b>699,635</b><span>Kaggle Playground S6E10</span></div>
<div class="dsb" style="right:96px;left:auto;background:#0a1f55"><div class="l">Real survey</div><b>129,880</b><span>Original survey dataset</span></div>
<div class="an hot" style="left:700px;top:290px;width:520px;text-align:center"><b>S→R</b>Transfer to real passengers</div>
<div class="an" style="left:700px;top:490px;width:520px;text-align:center"><b>R→S</b>Reverse transfer check</div>
<div class="an" style="left:96px;top:566px;width:520px"><b>S→S</b>Kaggle-style validation</div>
<div class="an" style="right:96px;top:566px;width:520px;text-align:right"><b>R→R</b>Reference performance</div>
<table class="spec"><tr><th>Cross-validation</th><th>Random seed</th><th>Settings</th><th>Metric</th></tr>
<tr><td>5-fold<span>stratified, fixed split per dataset</span></td><td>42<span>shared by every model</span></td><td>4<span>S→S, S→R, R→S, R→R</span></td><td>ROC AUC<span>higher is better</span></td></tr></table>
<div class="ftr">IS460 · Group 03 · Cleared for Satisfaction<b>3</b></div></div>'''
open('d4.html','w').write(page('Direction 4 · Corporate Scientific / Navy and Cyan',FONTS,CSS,[s1,s2,s3]))
print('d4 ok')
