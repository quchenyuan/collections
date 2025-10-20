---
layout: default
---

<h1>文档导航</h1>

{% comment %}
  Step 1: 获取所有 .md 页面（排除 index.md）
{% endcomment %}
{% assign all_md_pages = site.pages | where_exp: "page", "page.path contains '.md'" | where_exp: "page", "page.path != 'index.md'" | sort: "path" %}

{% comment %}
  Step 2: 提取所有目录路径
{% endcomment %}
{% assign all_dirs = "" | split: "" %}
{% for page in all_md_pages %}
  {% assign parts = page.path | split: '/' %}
  {% assign dir = "" %}
  {% for part in parts offset: 1 %}
    {% if forloop.last %}{% break %}{% endif %}
    {% if dir == "" %}
      {% assign dir = part %}
    {% else %}
      {% assign dir = dir | append: "/" | append: part %}
    {% endif %}
    {% unless all_dirs contains dir %}
      {% assign all_dirs = all_dirs | push: dir %}
    {% endunless %}
  {% endfor %}
{% endfor %}

{% comment %}
  Step 3: 排序并渲染目录标题（输出 HTML h2, h3...）
{% endcomment %}
{% assign sorted_dirs = all_dirs | sort: "size" %}
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
    {% assign level = depth | plus: 1 %}
    {% if level > 6 %}{% assign level = 6 %}{% endif %}
    {% capture tag %}h{{ level }}{% endcapture %}
    <{{ tag }}>{{ parts | last }}</{{ tag }}>
  {% endunless %}
{% endfor %}

{% comment %}
  Step 4: 渲染文件列表（带缩进）
{% endcomment %}
<ul>
{% for page in all_md_pages %}
  {% assign parts = page.path | split: '/' %}
  {% assign filename = parts | last | split: '.' | first %}
  {% assign depth = parts | size | minus: 2 %}
  {% if depth < 0 %}{% assign depth = 0 %}{% endif %}
  {% assign indent = "" %}
  {% for i in (1..depth) %}{% assign indent = indent | append: "&nbsp;&nbsp;" %}{% endfor %}
  <li>{{ indent | raw }}<a href="{{ page.url | relative_url }}">{{ filename }}</a></li>
{% endfor %}
</ul>
