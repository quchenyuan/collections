---
layout: default
---

# 文档导航

{% comment %}
  Step 1: 获取所有 .md 页面（排除 index.md）
{% endcomment %}
{% assign all_md_pages = site.pages | where_exp: "page", "page.path contains '.md'" | where_exp: "page", "page.path != 'index.md'" | sort: "path" %}

{% comment %}
  Step 2: 提取所有需要显示的目录路径（去重、排序）
{% endcomment %}
{% assign all_dirs = "" | split: "" %}
{% for page in all_md_pages %}
  {% assign parts = page.path | split: '/' %}
  {% assign current_path = "" %}
  {% for part in parts offset: 1 %}
    {% if forloop.last %}{% break %}{% endif %}
    {% if current_path == "" %}
      {% assign current_path = part %}
    {% else %}
      {% assign current_path = current_path | append: "/" | append: part %}
    {% endif %}
    {% unless all_dirs contains current_path %}
      {% assign all_dirs = all_dirs | push: current_path %}
    {% endunless %}
  {% endfor %}
{% endfor %}

{% comment %}
  Step 3: 按路径深度排序（确保父目录在子目录前）
{% endcomment %}
{% assign sorted_dirs = all_dirs | sort: "size" %}

{% comment %}
  Step 4: 渲染目录标题（使用 #, ##, ###... + 空行分隔）
{% endcomment %}
{% assign rendered_dirs = "" | split: "" %}
{% for dir in sorted_dirs %}
  {% assign parts = dir | split: '/' %}
  {% assign depth = parts | size %}
  {% assign skip = false %}
  {% for r in rendered_dirs %}
    {% if dir == r %}{% assign skip = true %}{% endif %}
  {% endfor %}
  {% unless skip %}
    {% assign rendered_dirs = rendered_dirs | push: dir %}
    {% assign hashes = "" %}
    {% for i in (1..depth) %}{% assign hashes = hashes | append: "#" %}{% endfor %}
{{ hashes }} {{ parts | last }}

{% endunless %}
{% endfor %}

{% comment %}
  Step 5: 渲染所有 .md 文件（带缩进）
{% endcomment %}
{% for page in all_md_pages %}
  {% assign parts = page.path | split: '/' %}
  {% assign filename = parts | last | split: '.' | first %}
  {% assign depth = parts | size | minus: 2 %}
  {% if depth < 0 %}{% assign depth = 0 %}{% endif %}
  {% assign indent = "" %}
  {% for i in (1..depth) %}{% assign indent = indent | append: "  " %}{% endfor %}
- {{ indent }}[{{ filename }}]({{ page.url | relative_url }})

{% endfor %}
