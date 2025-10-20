---
layout: default
---

# 文档导航

{% assign pages = site.pages | where_exp: "p", "p.path contains '.md'" | where_exp: "p", "p.path != 'index.md'" | sort: "path" %}

{% comment %}
  用于记录已经输出过的目录路径，避免重复
{% endcomment %}
{% assign rendered_dirs = "" | split: "" %}

{% for p in pages %}
  {% assign parts = p.path | split: '/' %}
  {% assign current_path = "" %}
  {% assign dir_output_needed = false %}
  {% assign dir_lines = "" | split: "" %}

  {% comment %}
    构建从根到文件所在目录的每一级路径，并检查是否已渲染
  {% endcomment %}
  {% for part in parts offset: 1 %}
    {% if forloop.last %}{% break %}{% endif %}
    {% if current_path == "" %}
      {% assign current_path = part %}
    {% else %}
      {% assign current_path = current_path | append: "/" | append: part %}
    {% endif %}

    {% unless rendered_dirs contains current_path %}
      {% assign rendered_dirs = rendered_dirs | push: current_path %}
      {% assign depth = current_path | split: '/' | size %}
      {% assign hashes = "" %}
      {% for i in (1..depth) %}{% assign hashes = hashes | append: "#" %}{% endfor %}
      {% assign line = hashes | append: " " | append: part %}
      {% assign dir_lines = dir_lines | push: line %}
    {% endunless %}
  {% endfor %}

  {% comment %}
    输出该文件路径中新增的目录标题（按顺序）
  {% endcomment %}
  {% for line in dir_lines %}
{{ line }}
  {% endfor %}

  {% comment %}
    输出文件本身（缩进 = 目录深度）
  {% endcomment %}
  {% assign filename = parts | last | split: '.' | first %}
  {% assign depth = parts | size | minus: 2 %}
  {% if depth < 0 %}{% assign depth = 0 %}{% endif %}
  {% assign indent = "" %}
  {% for i in (1..depth) %}{% assign indent = indent | append: "  " %}{% endfor %}
- {{ indent }}[{{ filename }}]({{ p.url | relative_url }})
{% endfor %}
