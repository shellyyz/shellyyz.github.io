---
layout: default
title: Personal
---

<section class="archive">
{% for personal in site.personal %}
{% unless personal.next %}

  {% unless forloop.first %}
    </div>
  </div>
  {% endunless %}

  <div class="archive-item fadeInDown animated">
    <h2>{{ personal.date | date: '%Y' }}</h2>
    <div>

{% else %}

{% capture year %}{{ personal.date | date: '%Y' }}{% endcapture %}
{% capture next_year %}{{ personal.next.date | date: '%Y' }}{% endcapture %}

{% if year != next_year %}

  {% unless forloop.first %}
    </div>
  </div>
  {% endunless %}

  <div class="archive-item fadeInDown animated">
    <h2>{{ personal.date | date: '%Y' }}</h2>
    <div>

{% endif %}
{% endunless %}

  <article>
    <a href="{{ personal.url | absolute_url }}" title="{{ personal.title }}">{{ personal.title }}</a>
    <div class="personal-date">
      <time datetime="{{ personal.date | date: '%Y-%m-%d' }}">{{ personal.date | date: "%-d %B" }}</time>
    </div>
  </article>

  {% if forloop.last %}
    </div>
  </div>
  {% endif %}

{% endfor %}
</section>
