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

## Statistical Forecasting (ARIMA)
- ARIMA model is fitted to the training data to capture time-series patterns.
- The model is tuned using auto_arima for optimal parameters.
- Forecasting is done for future time periods, and confidence intervals are calculated.
- The performance of the ARIMA model is evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

## Machine Learning Forecasting (LSTM)
- LSTM model architecture is designed for sequence prediction.
- TimeseriesGenerator is used to generate input-output pairs for the LSTM model.
- The model is trained on the training data using MSE loss and Adam optimizer.
- Model checkpoints are saved to track the best performing model during training.
- LSTM model is trained for multiple epochs to capture temporal dependencies.

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
