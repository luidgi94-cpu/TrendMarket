// Menu mobile + recherche locale (aucun suivi, aucun cookie)
(function () {
  var btn = document.querySelector('.menu-btn'), nav = document.getElementById('nav');
  if (btn && nav) btn.addEventListener('click', function () { var o = nav.classList.toggle('open'); btn.setAttribute('aria-expanded', o ? 'true' : 'false'); });
  var q = document.getElementById('q'), out = document.getElementById('results');
  if (!q || !out || !window.HGM_INDEX) return;
  function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]; }); }
  q.addEventListener('input', function () {
    var v = q.value.trim().toLowerCase();
    if (v.length < 2) { out.hidden = true; out.innerHTML = ''; return; }
    var r = window.HGM_INDEX.filter(function (x) { return (x.n + ' ' + x.d + ' ' + x.c).toLowerCase().indexOf(v) !== -1; }).slice(0, 12);
    out.hidden = false;
    out.innerHTML = r.length ? r.map(function (x) { return '<li><a href="' + esc(x.u) + '"><strong>' + esc(x.n) + '</strong><span>' + esc(x.c) + '</span></a></li>'; }).join('') : '<li style="padding:10px 12px">Aucun résultat. Essaie « tapis », « haltères » ou « élastiques ».</li>';
  });
  document.addEventListener('click', function (e) { if (!out.contains(e.target) && e.target !== q) out.hidden = true; });
})();
