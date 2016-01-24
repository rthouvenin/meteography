from django import template

register = template.Library()


@register.inclusion_tag('broadcaster/latest_prediction.html')
def latest_prediction(src):
    return {
        'name': src.name,
        'latest_prediction': src.latest_prediction,
    }
