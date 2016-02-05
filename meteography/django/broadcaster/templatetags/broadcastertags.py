from django import template

register = template.Library()


@register.inclusion_tag('broadcaster/latest_prediction.html')
def latest_prediction(src):
    return {
        'name': src.name,
        'latest_prediction': src.latest_prediction,
    }


@register.filter
def history(pred_params, orderby='-comp_date'):
    return pred_params.history(orderby=orderby)
