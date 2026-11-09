#!/bin/zsh
# usage: cd <TSA repo>/interactive && ../latex/check_interactive_page.sh chN_ro.html   (TSA; ported from MFM)
# needs a server on localhost:$PORT (default 8791) serving the TSA repo root: cd <TSA repo> && python3 -m http.server 8791
# (8765 is the MFM server; another port: PORT=8770 ../latex/check_...sh ...). Prints JS errors, empty readouts, number of plots, page width.
SP=${TMPDIR:-/tmp}; PORT=${PORT:-8791}
f=$1; t=_check_$RANDOM.html
python3 -c "
s=open('$f',encoding='utf-8').read().replace('</body>','<script>window.__err=[];addEventListener(\"error\",e=>window.__err.push(e.message));setTimeout(()=>{document.body.setAttribute(\"data-err\",JSON.stringify(window.__err)+\" empty=\"+[...document.querySelectorAll(\".out\")].filter(u=>!u.innerHTML).map(u=>u.id).join(\",\")+\" plots=\"+document.querySelectorAll(\".main-svg\").length+\" charts=\"+document.querySelectorAll(\".chart\").length+\" w=\"+innerWidth+\"/\"+document.documentElement.scrollWidth)},15000)</script></body>')
s=s.replace('<head>','<head><script>window.__err=[];addEventListener(\"error\",e=>(window.__err=window.__err||[]).push(e.message));</script>',1)
open('$t','w',encoding='utf-8').write(s)"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --disable-gpu --user-data-dir=$SP/chrome-prof-$RANDOM --window-size=1440,900 --virtual-time-budget=25000 --dump-dom "http://localhost:$PORT/interactive/$t" 2>/dev/null | grep -o 'data-err="[^"]*"'
python3 -c "import os; os.remove('$t')"
