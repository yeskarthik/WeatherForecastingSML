### TSLM and ARIMA functions

### TSLM Computation
### Input: FileName
### Output: Prediction Graph

process_TSLM<-function(filename)
{
  sea_weather<-read.csv(filename,stringsAsFactors = FALSE)
  weatherdf<-sea_weather
  weatherdf[is.na(weatherdf)]<-0
  weatherdf$DATE_HOUR<-as.POSIXct(weatherdf$DATE_HOUR)  ### Converting to date-time format
  train=subset.data.frame(weatherdf,weatherdf$DATE_HOUR<="2015-12-31")  ### Taking 2 years as Train
  test=subset.data.frame(weatherdf,weatherdf$DATE_HOUR>=as.POSIXct("2016-01-01")) ### Taking 1 year as Test 
  
  ### Creating Time Series values for train and test's dry bulb temperature
  temp_train<-ts(train$HOURLYDRYBULBTEMPF,frequency = 8760,start=c(2014,1)) 
  temp_test<-ts(test$HOURLYDRYBULBTEMPF,frequency = 8760,start=c(2016,1))
  plot.ts(temp_train)
  
  train<-train[-c(1,6,8,13)]
  test<-test[-c(1,6,8,13)]
  
  xreg_train<-data.frame(train[c(1:15)])
  xreg_test<-data.frame(test[c(1:15)])
  fit_ts<-tslm(temp_train~DAILYSUNRISE+DAILYSUNSET+HOURLYALTIMETERSETTING+HOURLYDEWPOINTTEMPF+HOURLYPRECIP+HOURLYPRESSURECHANGE+HOURLYPRESSURETENDENCY+HOURLYRELATIVEHUMIDITY+HOURLYSEALEVELPRESSURE+HOURLYSTATIONPRESSURE+HOURLYVISIBILITY+HOURLYWETBULBTEMPF+HOURLYWINDDIRECTION+HOURLYWINDGUSTSPEED+HOURLYWINDSPEED,data=train)
  fc_ts<-forecast(fit_ts,newdata=xreg_test)
  accuracy(fc_ts,temp_test)
  plot(fc_ts,main="Time Series Linear Regression Model (TSLM)- Prediction")### Blue lines indicate predictiom
  lines(temp_test,col="Red")### Red indicates test values
}

### ARIMA with extrenal Regressors Computation
### Input: FileName
### Output: Prediction Graph
process_ARIMA<-function(filename)
{
  sea_weather<-read.csv(filename,stringsAsFactors = FALSE)
  weatherdf<-sea_weather
  weatherdf[is.na(weatherdf)]<-0
  weatherdf$DATE_HOUR<-as.POSIXct(weatherdf$DATE_HOUR)  ### Converting to date-time format
  train=subset.data.frame(weatherdf,weatherdf$DATE_HOUR<="2015-12-31")  ### Taking 2 years as Train
  test=subset.data.frame(weatherdf,weatherdf$DATE_HOUR>=as.POSIXct("2016-01-01")) ### Taking 1 year as Test 
  
  ### Creating Time Series values for train and test's dry bulb temperature
  temp_train<-ts(train$HOURLYDRYBULBTEMPF,frequency = 8760,start=c(2014,1)) 
  temp_test<-ts(test$HOURLYDRYBULBTEMPF,frequency = 8760,start=c(2016,1))
  plot.ts(temp_train)
  
  train<-train[-c(1,6,8,13)]
  test<-test[-c(1,6,8,13)]
  
  xreg_train<-data.frame(train[c(1:15)])
  xreg_test<-data.frame(test[c(1:15)])
  #fit<-auto.arima(temp_train)
  fit_arima<-arima(temp_train,xreg=xreg_train,order=c(1,0,2))
  fc<-forecast(fit_arima,xreg=xreg_test)
  plot(fc,main="ARIMA with external Regressors- Prediction")
  accuracy(fc,temp_test)
  lines(temp_test,col="Red")### Red indicates test values
}