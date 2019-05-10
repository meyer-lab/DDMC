from IPython.display import HTML
import base64

def create_download_link(df, title = "Download CSV file", filename = "ClusterMembers_5cl_2Comp_MSR1.csv"):  
    csv = df.to_csv()
    b64 = base64.b64encode(csv.encode())
    payload = b64.decode()
    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'
    html = html.format(payload=payload,title=title,filename=filename)
    return HTML(html)

create_download_link(df)