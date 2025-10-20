---
layout: default
---

# 文档导航

{% assign pages = site.pages | where_exp: "p", "p.path contains '.md'" | where_exp: "p", "p.path != 'index.md'" | sort: "path" %}

{% comment %}
  收集所有目录路径（如 "GPU", "GPU/nvidia", "transformers/modules"）
{% endcomment %}
{% assign all_dirs = "" | split: "" %}
{% for p in pages %}
  {% assign parts = p.path | split: '/' %}
  {% assign current = "" %}
  {% for part in parts offset: 1 %}
    {% if forloop.last %}{% break %}{% endif %}
    {% if current == "" %}
      {% assign current = part %}
    {% else %}
      {% assign current = current | append: "/" | append: part %}
    {% endif %}
    {% unless all_dirs contains current %}
      {% assign all_dirs = all_dirs | push: current %}
    {% endunless %}
  {% endfor %}
{% endfor %}

{% comment %}
  按深度排序（确保父目录在子目录前）
{% endcomment %}
{% assign sorted_dirs = all_dirs | sort: "size" %}

{% comment %}
  渲染所有目录标题（用 #, ##, ###...）
{% endcomment %}
{% for dir in sorted_dirs %}
  {% assign parts = dir | split: '/' %}
  {% assign depth = parts | size %}
  {% assign hashes = "" %}
  {% for i in (1..depth) %}{% assign hashes = hashes | append: "#" %}{% endfor %}
{{ hashes }} {{ parts | last }}
{% endfor %}

{% comment %}
  渲染所有文件（按路径缩进）
{% endcomment %}
{% for p in pages %}
  {% assign parts = p.path | split: '/' %}
  {% assign filename = parts | last | split: '.' | first %}
  {% assign depth = parts | size | minus: 2 %}
  {% if depth < 0 %}{% assign depth = 0 %}{% endif %}
  {% assign indent = "" %}
  {% for i in (1..depth) %}{% assign indent = indent | append: "  " %}{% endfor %}
- {{ indent }}[{{ filename }}]({{ p.url | relative_url }})
{% endfor %}
