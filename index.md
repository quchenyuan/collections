# 文档导航

{% assign md_files = site.pages | sort: "path" | where_exp: "item", "item.path contains '.md'" %}
{% assign grouped_level1 = md_files | group_by_exp: "item", "item.path | split: '/' | first" %}

{% for level1 in grouped_level1 %}
  {% unless level1.name == "" or level1.name == "index.md" %}
# {{ level1.name }}

    {% assign subitems = level1.items | sort: "path" %}
    {% for file in subitems %}
      {% assign path_parts = file.path | split: '/' %}
      {% assign part_count = path_parts | size %}
      {% if part_count > 2 %}
        {% assign subgroup_name = path_parts[1] %}
      {% else %}
        {% assign subgroup_name = "" %}
      {% endif %}
      {% assign file.subgroup = subgroup_name %}
    {% endfor %}

    {% assign subgroups = subitems | group_by: "subgroup" %}
    {% for subgroup in subgroups %}
      {% if subgroup.name != "" %}
## {{ subgroup.name }}
      {% endif %}
      {% for file in subgroup.items %}
        {% assign filename_parts = file.path | split: '/' %}
        {% assign filename = filename_parts | last %}
        {% assign filename_no_ext = filename | split: '.' | first %}
- [{{ filename_no_ext }}]({{ file.url | relative_url }})
      {% endfor %}
    {% endfor %}
  {% endunless %}
{% endfor %}
