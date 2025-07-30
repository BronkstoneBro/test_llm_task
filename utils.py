from datetime import datetime, timedelta


def process_date_query(query):
    """Process date queries"""
    today = datetime.now().strftime("%d.%m.%Y")
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%d.%m.%Y")
    query = query.replace("сьогодні", today).replace("завтра", tomorrow)
    return query 