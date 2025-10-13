---
layout: default
title: 文档导航
---

# 文档导航

{% comment %}
获取所有 Markdown 文件（排除 index.md），按一级目录分组
{% endcomment %}
{% assign md_files = site.pages 
    | sort: "path" 
    | where_exp: "item", "item.path contains '.md'" 
%}
{% assign grouped_level1 = md_files | group_by_exp: "item", "item.path | split: '/' | first" %}

{% for level1 in grouped_level1 %}
  {% unless level1.name == "" or level1.name == "index.md" %}
# {{ level1.name }}

    {% comment %}
    按二级目录分组（如果有的话）
    {% endcomment %}
    {% assign grouped_level2 = level1.items | group_by_exp: "item", 
        "item.path | split: '/' | size > 2 
            ? item.path | split: '/' | slice: 1, 1 | first 
            : ''" 
    %}

    {% for level2 in grouped_level2 %}
      {% if level2.name != "" %}
## {{ level2.name }}
        {% assign subfiles = level2.items | sort: "path" %}
        {% for file in subfiles %}
          {% assign filename = file.path | split: '/' | last | split: '.' | first %}
- [{{ filename }}]({{ file.url | relative_url }})
        {% endfor %}
      {% else %}
        {% assign subfiles = level2.items | sort: "path" %}
        {% for file in subfiles %}
          {% assign filename = file.path | split: '/' | last | split: '.' | first %}
- [{{ filename }}]({{ file.url | relative_url }})
        {% endfor %}
      {% endif %}
    {% endfor %}
  {% endunless %}
{% endfor %}

