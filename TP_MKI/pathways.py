class Pathways:
    
    def videos(self):
        videos = "C:/Users/hanna/Documents/Thesis/videos/"
        return videos
    
    def home_videos_performance(self):
        mm_cad1 = "C:/Users/hanna/Documents/Thesis/results/mm_cad1_live_output.csv"
        mn_cad1 = "C:/Users/hanna/Documents/Thesis/results/mn_cad1_live_output.csv"
        ms_cad1 = "C:/Users/hanna/Documents/Thesis/results/ms_cad1_live_output.csv"
        ms_cad2 = "C:/Users/hanna/Documents/Thesis/results/ms_cad2_live_output.csv"
        ms_cad3 = "C:/Users/hanna/Documents/Thesis/results/ms_cad3_live_output.csv"
        return [mm_cad1, mn_cad1, ms_cad1, ms_cad2, ms_cad3]
    
    def home_videos_exp(self):
        mm_cad1 = "C:/Users/hanna/Documents/Thesis/videos/mm_cad1_live_output/"
        mn_cad1 = "C:/Users/hanna/Documents/Thesis/videos/mn_cad1_live_output/"
        ms_cad1 = "C:/Users/hanna/Documents/Thesis/videos/ms_cad1_live_output/"
        ms_cad2 = "C:/Users/hanna/Documents/Thesis/videos/ms_cad2_live_output/"
        return [mm_cad1, mn_cad1, ms_cad1, ms_cad2]
    
    def home_videos_no_touch(self):
        ms_cad3 = "C:/Users/hanna/Documents/Thesis/videos/ms_cad3_live_output/"
        return [ms_cad3]

pathways = Pathways()