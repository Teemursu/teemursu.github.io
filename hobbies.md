---
layout: default
---

<iframe id="sc-player" width="100%" height="300" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/soundcloud%253Atracks%253A2108375115&color=%23d19c90&auto_play=false&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe>
<div style="font-size: 10px; color: #cccccc; line-break: anywhere; word-break: normal; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; font-family: Interstate, Lucida Grande, Lucida Sans Unicode, Lucida Sans, Garuda, Verdana, Tahoma, sans-serif; font-weight: 100;">
  <a href="https://soundcloud.com/polynoises" title="polynoises" target="_blank" style="color: #cccccc; text-decoration: none;">polynoises</a> ·
  <a href="https://soundcloud.com/polynoises/post-cosmos" title="Post Cosmos" target="_blank" style="color: #cccccc; text-decoration: none;">Post Cosmos</a>
</div>

<div class="video-embeds">
  <iframe id="yt-one" width="100%" height="315" src="https://www.youtube.com/embed/3CuYk55q_ec?enablejsapi=1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
  <iframe id="yt-two" width="100%" height="315" src="https://www.youtube.com/embed/FNW21KpH33c?enablejsapi=1" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

{% assign lofoten = site.static_files | where_exp: "file", "file.path contains '/assets/hobbies/lofoten/'" | sort: "path" %}
<div class="gallery">
  {% for photo in lofoten %}
  <img src="{{ photo.path | relative_url }}" alt="Lofoten photo">
  {% endfor %}
</div>

<div id="lightbox" class="lightbox hidden">
  <img id="lightbox-img" src="" alt="Expanded view">
</div>

<script src="https://w.soundcloud.com/player/api.js"></script>
<script src="https://www.youtube.com/iframe_api"></script>
<script>
  (function() {
    let scWidget;
    const scFrame = document.getElementById('sc-player');
    if (scFrame && window.SC && SC.Widget) {
      scWidget = SC.Widget(scFrame);
    } else {
      window.addEventListener('load', function() {
        if (window.SC && SC.Widget) scWidget = SC.Widget(scFrame);
      });
    }

    const ytIds = ['yt-one', 'yt-two'];
    let ytPlayers = [];
    window.onYouTubeIframeAPIReady = function() {
      ytPlayers = ytIds.map(function(id) {
        const el = document.getElementById(id);
        if (!el) return null;
        return new YT.Player(id, {
          events: {
            'onStateChange': function(event) {
              if (event.data === YT.PlayerState.PLAYING && scWidget) {
                scWidget.pause();
              }
            }
          }
        });
      }).filter(Boolean);
    };

    const lightbox = document.getElementById('lightbox');
    const lightboxImg = document.getElementById('lightbox-img');
    if (lightbox && lightboxImg) {
      document.querySelectorAll('.gallery img').forEach(function(img) {
        img.addEventListener('click', function() {
          lightboxImg.src = img.src;
          lightbox.classList.remove('hidden');
        });
      });
      lightbox.addEventListener('click', function() {
        lightbox.classList.add('hidden');
        lightboxImg.src = '';
      });
      document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape') {
          lightbox.classList.add('hidden');
          lightboxImg.src = '';
        }
      });
    }
  })();
</script>
