from private_keys import api_key , id_num
from apiclient.discovery import build

class Handler():

    def __init__(self):
        # initialize a connection to the api
        self.resource = build("customsearch", 'v1', developerKey=api_key).cse()


    def get_imgs_url(self, query = 'bread street kitchen in Singapore' , num_of_img = 10):
        list_of_imgs = []
        result = self.resource.list(q=query, cx=id_num, searchType='image').execute()
        for item in result['items']:
            list_of_imgs.append(item['link'])

        return list_of_imgs[:num_of_img]


def main():
    img_hand = Handler()
    user_query = input("enter your query in order to get its related images")
    result = img_hand.get_imgs_url(query = user_query, num_of_img = 5)
    print(result)

if __name__ == "__main__":
    main()




# resource = build("customsearch", 'v1', developerKey=api_key).cse()
#
# def imgs_url_get(query = 'bread street kitchen in Singapore', num_of_img = 10):
#     list_of_imgs = []
#     result = resource.list(q=query, cx=id_num, searchType='image').execute()
#     for item in result['items']:
#         list_of_imgs.append(item['link'])
#
#     print(list_of_imgs)
#
# ''' to sea a pretified json file, visit:
# https://jsonformatter.curiousconcept.com/#
# '''
