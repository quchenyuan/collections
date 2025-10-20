---
layout: default
---
# 文档导航

{% assign md_files = site.pages | where_exp: "item", "item.path contains '.md'" %}
{% assign md_files = md_files | reject: "path", "index.md" %}
{% assign sorted_files = md_files | sort: "path" %}

{% assign tree = {} %}

{% for file in sorted_files %}
  {% assign parts = file.path | split: '/' %}
  {% assign current_node = tree %}

  {% for part in parts %}
    {% assign is_last = forloop.last %}

    {% if is_last %}
      {% assign current_files = current_node["__files"] | default: "" | split: "" %}
      {% assign current_files = current_files | push: file %}
      {% assign current_node = current_node | merge: {"__files": current_files} %}
    {% else %}
      {% if current_node[part] == nil %}
        {% assign current_node = current_node | merge: { part: {} } %}
      {% endif %}
      {% assign current_node = current_node[part] %}
    {% endif %}
  {% endfor %}
{% endfor %}

{% include print_tree.liquid tree=tree indent=1 %}

