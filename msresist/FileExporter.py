""" Downloads generated csv files. """

import base64
from IPython.display import HTML


def create_download_link(df, filename, title="Download CSV file"):
    "Download a csv file from a pandas dataframe."
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload, title=title, filename=filename)
    return HTML(html)
