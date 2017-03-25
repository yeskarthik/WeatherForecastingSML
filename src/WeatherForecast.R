
getData<- function() 
{
  headers<-("TimeWITA,TemperatureF,Dew PointF,Humidity,Sea Level PressureIn,VisibilityMPH,Wind Direction,Wind SpeedMPH,Gust SpeedMPH,PrecipitationIn,Events,Conditions,WindDirDegrees,DateUTC")
  weather_table<-read.table(textConnection(headers), sep = ",")
  dates<-seq(as.Date("2015/1/1"), as.Date("2017/1/1"),by=1)
  for (d in 1:length(dates))
  {
    date<-gsub("-","/",dates[d])
    url<-paste("http://www.wunderground.com/history/airport/SEA/",date,sep="")
    url<-paste(url,"/DailyHistory.html?format=1",sep="")
    thepage <-readLines(url)
    thepage<-gsub("<br />","",thepage)
    wdf<-read.table(textConnection(thepage), sep = ",")
    weather_table<-rbind(weather_table,wdf[-1,])
  }
  weather_table<-weather_table[,c(ncol(weather_table), 1:(ncol(weather_table)-1))]
  return (weather_table)
}
weatherdf<-data.frame(getData())
########################################
# library(forecast)
# fweather<-forecast(weatherdf)
# autoplot(fweather)