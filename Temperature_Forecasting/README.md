# Bhubaneswar Surface Temperature Forecasting

This project aims to forecast surface temperatures in Bhubaneswar using historical temperature data. The project utilizes various techniques, including statistical methods like ARIMA (AutoRegressive Integrated Moving Average) and machine learning algorithms like LSTM (Long Short-Term Memory) networks. The forecasted temperatures can be valuable for urban planning, agriculture, and weather-dependent industries.

## Dataset
The dataset used in this project contains historical surface temperature data for Bhubaneswar. It includes monthly average temperature readings over a certain period.

### Data Preprocessing
- Data is cleaned to handle missing values and outliers.
- Outliers are identified using z-score and IQR methods.
- Missing values are filled using mean imputation.
- The data is then resampled to monthly frequency and missing values are dropped.

## Exploratory Data Analysis (EDA)
- Box plots, histograms, and scatter plots are used to visualize the distribution of temperature data.
- Rolling mean is calculated to observe trends in temperature over time.
- Seasonal decomposition is performed to analyze trend, seasonality, and residuals.

## Statistical Forecasting (SARIMAX)
![image](https://github.com/Sayakhatui/Time-series-Analysis-of-Temperature/assets/150340995/7e512dca-c1b6-47e6-ad78-a87d51413fd2)

- SARIMAX model is fitted to the training data to capture time-series patterns.
- The model is tuned using auto_arima for optimal parameters.
- Forecasting is done for future time periods, and confidence intervals are calculated.
- The performance of the ARIMA model is evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

## Machine Learning Forecasting (LSTM)
![image](https://github.com/Sayakhatui/Time-series-Analysis-of-Temperature/assets/150340995/954e40a2-3f2e-413b-a184-dd1a2a6a4db1)

- LSTM model architecture is designed for sequence prediction.
- TimeseriesGenerator is used to generate input-output pairs for the LSTM model.
- The model is trained on the training data using MSE loss and Adam optimizer.
- Model checkpoints are saved to track the best performing model during training.
- LSTM model is trained for multiple epochs to capture temporal dependencies.
  
## SARIMAX vs LSTM Model Performance ##  
![image](https://github.com/Sayakhatui/Time-series-Analysis-of-Temperature/assets/150340995/77601864-ccf8-41ff-95ce-2f2ed47b65b9)
**<u>SARIMAX Model</u>**
The SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous regressors) model is a linear statistical model that is widely used for time series forecasting. It incorporates both autoregression (AR) and moving average (MA) elements, and can handle trends and seasonality in the data.
In this case, the SARIMAX model achieved an RMSE (Root Mean Square Error) value of **0.9**. The RMSE is a measure of the differences between the values predicted by the model and the actual values. Lower RMSE values indicate better fit. **Therefore, an RMSE of 0.9 suggests that the SARIMAX model has performed quite well**.
**<u>LSTM Model</u>**
On the other hand, the LSTM (Long Short-Term Memory) model is  a type of recurrent neural network (RNN) that is capable of learning long-term dependencies, which makes it suitable for time series forecasting.
     However, in your case, the LSTM model achieved an RMSE value of **1.14**, which is higher than that of the SARIMAX model. **This suggests that the LSTM model’s predictions were, on average, slightly less accurate than those of the SARIMAX model.**
    
## Streamlit Implementation
- The model has been implemented into a Streamlit web application (`app.py`).
- Users can input a date to get a temperature prediction.
- Users can also input a range of dates to plot the temperature vs. time graph.

## Results
- Forecasted temperatures from both ARIMA and LSTM models are visualized and compared with observed temperatures.
- Performance metrics such as MSE and RMSE are calculated for both models.
- The accuracy and efficiency of each model are discussed, along with potential areas of improvement.

## Dependencies
- numpy
- pandas
- matplotlib
- seaborn
- statsmodels
- pmdarima
- scikit-learn
- tensorflow
- keras
- streamlit

## Project Structure
- `data.csv`: Input dataset containing historical temperature data.
- `app.py`: Streamlit web application for temperature forecasting.
- `Temperature.ipynb`: Jupyter notebook containing the project code.
- `README.md`: Project documentation.

## Instructions
1. Ensure all dependencies are installed.
2. Run the Streamlit web application by executing `streamlit run app.py`.
3. Follow the instructions provided in the web application to input date(s) and obtain temperature prediction(s) or plot temperature vs. time graph.
4. Experiment with different input dates and date ranges to explore temperature forecasts.
5. For more detailed exploration and model development, refer to the `Temperature.ipynb` notebook.
6. For better understanding, refer to the 'Time-Series-Analysis-of-bbsr-Temperature.pptx' powerpoint presentation.
