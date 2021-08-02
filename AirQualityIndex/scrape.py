#link to tutorial : https://www.youtube.com/watch?v=CTu0qnuMxgA&list=PLZoTAELRMXVOFnfSwkB_uyr4FT-327noK

import time
import os
import requests
import sys
def retrieve_html():
    for year in range(2013,2019):
        for month in range(1,13):
            print(year,month)
            if month < 10:
                url =  'http://en.tutiempo.net/climate/0{}-{}/ws-421820.html'.format(month,year)

            else:
                url =  'http://en.tutiempo.net/climate/{}-{}/ws-421820.html'.format(month,year)

            text = requests.get(url)

            text_utf = text.text.encode('utf-8')

            if not os.path.exists('Data/Html_Data/{}'.format(year)):
                os.makedirs('Data/Html_Data/{}'.format(year))

            with open("Data/Html_Data/{}/{}.html".format(year, month),'wb') as output:
                output.write(text_utf)
        
        sys.stdout.flush()
        

if __name__ == "__main__":
    start_time = time.time()
    retrieve_html()
    end_timne = time.time()
    print("Exe time => {}".format(end_timne - start_time))

