---
layout: default
---
# 文档导航

{% assign md_files = site.static_files | where_exp: "item", "item.path ends_with '.md'" %}
{% assign md_files = md_files | sort: "path" %}

{% assign current_path_parts = "" | split: "" %}

{% for file in md_files %}
  {% assign parts = file.path | split: '/' %}
  {% assign depth = parts | size %}

  {%- comment -%}
    遍历每级目录，如果和上一个文件的路径不同就输出标题
  {%- endcomment -%}
  {% for i in (0..depth-2) %}
    {% assign dir_name = parts[i] %}
    {% if dir_name != current_path_parts[i] %}
      {% assign heading_level = i | plus: 1 %}
{% for j in (1..heading_level) %}#{% endfor %} {{ dir_name }}
    {% endif %}
  {% endfor %}

  {% assign current_path_parts = parts %}

  {% assign filename = parts | last %}
  {% assign filename_no_ext = filename | split: '.' | first %}
- [{{ filename_no_ext }}]({{ file.path | relative_url }})
{% endfor %}

