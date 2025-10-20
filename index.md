---
layout: default
---

# 文档导航

{% assign pages = site.pages | where_exp: "p", "p.path contains '.md'" | where_exp: "p", "p.path != 'index.md'" | sort: "path" %}

{% comment %}收集所有目录路径{% endcomment %}
{% assign dirs = "" | split: "" %}
{% for p in pages %}
  {% assign parts = p.path | split: '/' %}
  {% assign path = "" %}
  {% for part in parts offset: 1 %}
    {% if forloop.last %}{% break %}{% endif %}
    {% if path == "" %}
      {% assign path = part %}
    {% else %}
      {% assign path = path | append: "/" | append: part %}
    {% endif %}
    {% unless dirs contains path %}
      {% assign dirs = dirs | push: path %}
    {% endunless %}
  {% endfor %}
{% endfor %}

{% comment %}按深度排序并输出标题{% endcomment %}
{% assign sorted_dirs = dirs | sort: "size" %}
{% assign seen = "" | split: "" %}
{% for dir in sorted_dirs %}
  {% assign parts = dir | split: '/' %}
  {% assign depth = parts.size %}
  {% unless seen contains dir %}
    {% assign seen = seen | push: dir %}
    {% assign hashes = "" %}
    {% for i in (1..depth) %}{% assign hashes = hashes | append: "#" %}{% endfor %}
{{ hashes }} {{ parts.last }}
  {% endunless %}
{% endfor %}

{% comment %}输出文件{% endcomment %}
{% for p in pages %}
  {% assign parts = p.path | split: '/' %}
  {% assign name = parts.last | split: '.' | first %}
  {% assign depth = parts.size | minus: 2 %}
  {% if depth < 0 %}{% assign depth = 0 %}{% endif %}
  {% assign indent = "" %}
  {% for i in (1..depth) %}{% assign indent = indent | append: "  " %}{% endfor %}
- {{ indent }}[{{ name }}]({{ p.url | relative_url }})
{% endfor %}
