from private_keys import api_key , id_num
from apiclient.discovery import build
resource = build("customsearch", 'v1', developerKey=api_key).cse()


def imgs_url_get(query = 'bread street kitchen in Singapore', num_of_img = 10):
    list_of_imgs = []
    result = resource.list(q=query, cx=id_num, searchType='image').execute()
    for item in result['items']:
        list_of_imgs.append(item['link'])

    print(list_of_imgs)

''' to sea a pretified json file, visit:
https://jsonformatter.curiousconcept.com/#
'''
