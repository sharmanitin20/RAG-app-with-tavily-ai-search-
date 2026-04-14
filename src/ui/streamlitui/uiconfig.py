from configparser import ConfigParser


class Config:
    def __init__(self,config_file="./src/ui/streamlitui/uiconfig.ini"):
        self.config = ConfigParser()
        self.config.read(config_file)

    def get_page_title(self):
        return self.config["DEFAULT"]["PAGE_TITLE"]  
     
    def get_page_caption(self):
        return self.config["DEFAULT"]["PAGE_CAPTION"]