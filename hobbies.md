---
layout: default
---

<iframe width="100%" height="300" scrolling="no" frameborder="no" allow="autoplay" src="https://w.soundcloud.com/player/?url=https%3A//api.soundcloud.com/tracks/soundcloud%253Atracks%253A2108375115&color=%23d19c90&auto_play=true&hide_related=false&show_comments=true&show_user=true&show_reposts=false&show_teaser=true&visual=true"></iframe>
<div style="font-size: 10px; color: #cccccc; line-break: anywhere; word-break: normal; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; font-family: Interstate, Lucida Grande, Lucida Sans Unicode, Lucida Sans, Garuda, Verdana, Tahoma, sans-serif; font-weight: 100;">
  <a href="https://soundcloud.com/polynoises" title="polynoises" target="_blank" style="color: #cccccc; text-decoration: none;">polynoises</a> ·
  <a href="https://soundcloud.com/polynoises/post-cosmos" title="Post Cosmos" target="_blank" style="color: #cccccc; text-decoration: none;">Post Cosmos</a>
</div>

{% assign lofoten = site.static_files | where_exp: "file", "file.path contains '/assets/hobbies/lofoten/'" | sort: "path" %}
<div class="gallery">
  {% for photo in lofoten %}
  <img src="{{ photo.path | relative_url }}" alt="Lofoten photo">
  {% endfor %}
</div>
