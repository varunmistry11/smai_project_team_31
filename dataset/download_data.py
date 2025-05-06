import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

#mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory=".")
mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory="./task")
mySoccerNetDownloader.password = "s0cc3rn3t"

#mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv", "Labels-v2.json", "Labels-cameras.json", "Labels-replays.json", "tracking_data.json"], split=["train", "valid", "test"])

# Download SoccerNet v3 data
#mySoccerNetDownloader.downloadGames(version=3, task="tracking")  # Player tracking data
#mySoccerNetDownloader.downloadGames(version=3, task="multi-view")  # Multi-view camera data
#mySoccerNetDownloader.downloadGames(version=3, task="ball-tracking")  # Ball tracking data

#mySoccerNetDownloader.downloadDataTask(task="spotting-2023", split=["train", "valid", "test", "challenge"])
mySoccerNetDownloader.downloadDataTask(task="spotting-ball-2023", split=["train", "valid", "test", "challenge"], password="s0cc3rn3t")
