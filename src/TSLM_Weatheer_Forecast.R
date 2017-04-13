### Preprocessed dataset and TSLM Modelling

### Description: Converted Column 1 to Date Time Format and aggregated hours to day. Removed muti valueed Columns 8 and 13.  
get_daily_data<-function()
{
  weatherdf$DATE_HOUR<-as.POSIXct(sea_weather$DATE_HOUR)
  weatherdf2<-as.data.frame(seq(as.POSIXct("2014-01-01"), as.POSIXct("2016-12-31"), "day"))
  colnames(weatherdf2)<-"DATE_HOUR"
  wcolnames<-names(weatherdf)
  for (c in 2:length(wcolnames))
  {
    if (c!=8 && c!=13)
    {
      ag <- aggregate(weatherdf[[c]] ~ DATE_HOUR, weatherdf,FUN=mean)
      weatherdf2<-merge(weatherdf2,ag)
      colnames(weatherdf2)[ncol(weatherdf2)]<-wcolnames[c]
    }
  }
  
  
  return (weatherdf2)
}

sea_weather<-read.csv("C:/Users/shwetha/Documents/SML/Weather Forecast/WeatherForecastingSML/dataset/processed_SEA.csv",stringsAsFactors = FALSE)
weatherdf<-sea_weather
weatherdf[is.na(weatherdf)]<-0
weatherdf<-get_daily_data()

weatherdf$DATE_HOUR<-as.POSIXct(weatherdf$DATE_HOUR)  ### Converting to date-time format
train=subset.data.frame(weatherdf,weatherdf$DATE_HOUR<="2015-12-31")  ### Taking 2 years as Train
test=subset.data.frame(weatherdf,weatherdf$DATE_HOUR>=as.POSIXct("2016-01-01")) ### Taking 1 year as Test 


library(forecast)
### Creating Time Series values for train and test's dry bulb temperature
temp_train<-ts(train$HOURLYDRYBULBTEMPF,frequency = 365.25,start=c(2014,1)) 
temp_test<-ts(test$HOURLYDRYBULBTEMPF,frequency = 365.25,start=c(2016,1))
plot.ts(temp_train)


### Building linearmodel with all parameters as input
fit<-lm(temp_train~train$HOURLYPRECIP+train$HOURLYWETBULBTEMPF+train$HOURLYPRESSURECHANGE+train$DAILYSUNRISE+train$DAILYSUNSET+train$HOURLYALTIMETERSETTING+train$HOURLYPRESSURETENDENCY+train$HOURLYRELATIVEHUMIDITY+train$HOURLYSEALEVELPRESSURE+train$HOURLYSTATIONPRESSURE+train$HOURLYVISIBILITY+train$HOURLYWINDDIRECTION+train$HOURLYWINDGUSTSPEED+train$HOURLYWINDSPEED)
ols<-step(fit,directio="both")  ### Applied stepwise regression on the linear model for feature selection

### Identified parameters that affect using Stepwise Regression
### Note: Stepwose Regression gives more features. Yet to update the model
### Applying TSLM model
fit_ts<-tslm(temp_train~HOURLYPRECIP+HOURLYWETBULBTEMPF+HOURLYPRESSURECHANGE,data=train)

### Comparing output of Stepwise Regression and Linear model fit
summary(fit)
summar(ols)

### Forecasting Results of TSLM model where TestNewData is the feature values for 2016
test_newdata<-subset.data.frame(test,select=c("HOURLYPRECIP","HOURLYWETBULBTEMPF","HOURLYPRESSURECHANGE"))
fc_ts<-forecast(fit_ts,newdata=test_newdata)

plot(fc_ts)### Blue lines indicate predictiom
lines(temp_test,col="Red")### Red indicates test values


