function selectSession(id) {
  document.querySelectorAll('.session-view').forEach(v => v.style.display = 'none');
  document.querySelectorAll('.session-item').forEach(v => v.classList.remove('active'));

  const view = document.getElementById('view-' + id);
  const nav  = document.getElementById('nav-'  + id);
  if (view) view.style.display = 'block';
  if (nav)  nav.classList.add('active');
}

function toggleClip(sessionId, clipId) {
  const card = document.getElementById('card-' + sessionId + '-' + clipId);
  if (card) card.classList.toggle('open');
}
