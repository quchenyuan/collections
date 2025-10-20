---
layout: default
---

# 文档导航

{% comment %}
  Step 1: 获取所有 .md 页面（排除首页和非内容页）
{% endcomment %}
{% assign all_md_pages = site.pages | where_exp: "page", "page.path contains '.md'" | where_exp: "page", "page.path != 'index.md'" %}

{% comment %}
  Step 2: 提取所有唯一目录路径（仅包含有 .md 文件的目录）
{% endcomment %}
{% assign all_dirs = "" | split: "" %}
{% for page in all_md_pages %}
  {% assign path_parts = page.path | split: '/' %}
  {% assign dir_path = "" %}
  {% for part in path_parts offset: 1 %}
    {% if forloop.last %}
      {% break %}
    {% endif %}
    {% if dir_path == "" %}
      {% assign dir_path = part %}
    {% else %}
      {% assign dir_path = dir_path | append: "/" | append: part %}
    {% endif %}
    {% unless all_dirs contains dir_path %}
      {% assign all_dirs = all_dirs | push: dir_path %}
    {% endunless %}
  {% endfor %}
{% endfor %}

{% comment %}
  Step 3: 按层级排序目录（确保父目录在子目录前）
{% endcomment %}
{% assign sorted_dirs = all_dirs | sort: "size" %}

{% comment %}
  Step 4: 渲染目录树（模拟递归）—— 使用 raw 防止转义
{% endcomment %}
{% assign rendered_dirs = "" | split: "" %}

{% for dir in sorted_dirs %}
  {% assign dir_parts = dir | split: '/' %}
  {% assign depth = dir_parts | size %}
  {% assign is_rendered = false %}
  {% for rendered in rendered_dirs %}
    {% if dir == rendered %}
      {% assign is_rendered = true %}
    {% endif %}
  {% endfor %}
  {% if is_rendered != true %}
    {% assign rendered_dirs = rendered_dirs | push: dir %}

    {% comment %}计算缩进或标题级别{% endcomment %}
    {% assign heading_level = depth | plus: 1 %}
    {% if heading_level > 6 %}{% assign heading_level = 6 %}{% endif %}
    <h{{ heading_level }}>{{ dir_parts | last }}</h{{ heading_level }}>{% comment %}用 raw 输出{% endcomment %}
  {% endif %}
{% endfor %}

{% comment %}
  Step 5: 渲染所有 .md 文件，按路径缩进显示
{% endcomment %}
{% assign sorted_pages = all_md_pages | sort: "path" %}
{% for page in sorted_pages %}
  {% assign path_parts = page.path | split: '/' %}
  {% assign filename = path_parts | last | split: '.' | first %}
  {% assign dir_path = "" %}
  {% for part in path_parts offset: 1 %}
    {% if forloop.last %}
      {% break %}
    {% endif %}
    {% if dir_path == "" %}
      {% assign dir_path = part %}
    {% else %}
      {% assign dir_path = dir_path | append: "/" | append: part %}
    {% endif %}
  {% endfor %}

  {% comment %}计算缩进（根据目录深度）{% endcomment %}
  {% assign depth = path_parts | size | minus: 2 %}
  {% if depth < 0 %}{% assign depth = 0 %}{% endif %}
  {% assign indent = "" %}
  {% for i in (1..depth) %}
    {% assign indent = indent | append: "  " %}
  {% endfor %}

- {{ indent }}[{{ filename }}]({{ page.url | relative_url }})
{% endfor %}
