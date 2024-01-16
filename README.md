# Group_Project_4-NFL_Game_Outcome_Predictor
Group project 4 using machine learning.

////////////////////////////////////////////
NFL Machine Learning Model
////////////////////////////////////////////

Summary:
--------------------------------------------------
For summmary and findings, see "Presentation.pptx" file


////////////////////////////////////////////
Sources for Data
////////////////////////////////////////////


The folowing data sources were used:


Wikipeia Pages:

The game and statistics data was pulled from Wikipedia.  Since we were concentrating on the 2022 season, the url for each team was in the following format:

    https://en.wikipedia.org/wiki/2022_{team_name}_season

We developed a list of team names and performed the data retrieval in a `for` loop.

Images:

The NFL logo and team images used in the prediction application were manually downloaded and stored with the application from this url:

    https://www.nfl.com/teams/

////////////////////////////////////////////
Sources for Code
////////////////////////////////////////////


--------------------------------------------------
Grabbing_Data.ipynb
--------------------------------------------------

The following source was used for the method of pulling the data
from Wikipedia tables:

    https://pbpython.com/pandas-html-table.html


--------------------------------------------------
Wrangling_Data.ipynb
--------------------------------------------------

From:
    
    https://www.geeksforgeeks.org/how-to-drop-rows-that-contain-a-specific-string-in-pandas/

The following method was used:

    df = df[df["team"].str.contains("Team 1") == False]

To write:

    cur_df = cur_df[cur_df["Result"].str.contains("Bye") == False]

and:
    cur_df_cleaned_home = cur_df_cleaned[cur_df_cleaned["Opponent"].str.contains("at ") == False]

and:
    cur_df_cleaned_away = cur_df_cleaned[cur_df_cleaned["Opponent"].str.contains("at ") == True]


From:

    https://saturncloud.io/blog/how-to-delete-rows-with-null-values-in-a-specific-column-in-pandas-dataframe/#:~:text=Deleting%20rows%20with%20null%20values%20in%20a%20specific%20column%20can,values%20in%20the%20specified%20column.&text=df%20is%20the%20Pandas%20DataFrame%20that%20you%20want%20to%20modify.

The following method was used:

    # Drop Games With Null Values for Time
    cur_df = cur_df.dropna(subset=["time"], how="any")


From:

    Professor Benjamin Alford

The method for the concatination of dataframes was used to write:

    master_home_df = pd.concat([master_home_df, cur_df_cleaned_home], ignore_index= True)

and:

    master_away_df = pd.concat([master_away_df, cur_df_cleaned_away], ignore_index= True)

--------------------------------------------------
Building_Model.ipynb
--------------------------------------------------

The Following Section:

    # Mount Drive
    from google.colab import drive
    drive.mount('/content/drive')

Was provided by Google Colab when you select "Mount Drive"


The Following Section:

    # Create a StandardScaler Instance
    scaler = StandardScaler()

    # Fit the StandardScaler
    features_scaler = scaler.fit(features_array)

    # Scale the Features
    scaled_features = features_scaler.transform(features_array)

Was taken from Caleb Wolf's "AlphabetSoupCharity_Optimization.ipynb" file from the week 21 homework using the method provided in the starter file:

    # Create a StandardScaler Instance
    scaler = StandardScaler()

    # Fit the StandardScaler
    features_scaler = scaler.fit(features_array)

    # Scale the Features
    scaled_features = features_scaler.transform(features_array)


For the Following Section:

    # Create Primary Component Analysis Model With 95% Explainability
    pca_model=PCA(n_components=0.95)

The method of PCA using a % Variance instead of a number came from the following source:

    https://mikulskibartosz.name/pca-how-to-choose-the-number-of-components


The Following Section:

    rf_model = RandomForestClassifier(random_state=1, n_estimators=1000).fit(X_train, y_train)

    print(f'Training Score: {rf_model.score(X_train, y_train)}')
    print(f'Testing Score: {rf_model.score(X_test, y_test)}')

Was mirroring code from Week 20, Class 3, Activity 4, "04-Ins_Forest-Features", "RandomForest-Feature-Selection.ipynb" provided by Professor Benajamin Alfrord :

    clf = RandomForestClassifier(random_state=1, n_estimators=500).fit(X_train_scaled, y_train)
    print(f'Training Score: {clf.score(X_train_scaled, y_train)}')
    print(f'Testing Score: {clf.score(X_test_scaled, y_test)}')

This was also used for analyzing effectiveness of other sections with nearly identical syntax.

The following Section:


Used the method from Week 20, Class 2, Activity 5, "05-Ins_KNN", Ins_K_Nearest_Neighbors:

    # Loop through different k values to find which has the highest accuracy.
    # Note: We use only odd numbers because we don't want any ties.
    train_scores = []
    test_scores = []
    for k in range(1, 20, 2):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_score = knn.score(X_train, y_train)
        test_score = knn.score(X_test, y_test)
        train_scores.append(train_score)
        test_scores.append(test_score)
        print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")
        
        
    plt.plot(range(1, 20, 2), train_scores, marker='o')
    plt.plot(range(1, 20, 2), test_scores, marker="x")
    plt.xlabel("k neighbors")
    plt.ylabel("Testing accuracy Score")
    plt.show()

For the following section:

# Check for K-Value with The Heighest Accuracy
    training_data_scores = []
    testing_data_scores = []

    for k in range(1, 50, 2):

        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train, y_train)

        train_score = knn_model.score(X_train, y_train)
        test_score = knn_model.score(X_test, y_test)

        training_data_scores.append(train_score)
        testing_data_scores.append(test_score)

        print(f"k: {k}, Train/Test Score: {train_score:.3f}/{test_score:.3f}")
        
        
    plt.plot(range(1, 50, 2), training_data_scores, marker='o')
    plt.plot(range(1, 50, 2), testing_data_scores, marker="x")
    plt.xlabel("Number of Neighbors")
    plt.ylabel("Testing Accuracy Score")
    plt.show()

The code for the Nural Network Model:

    # Count features
    features_count = len(X_train[0])
    print(f'Total Features: {features_count}')

    # Define Nural Network Model

    nn = tf.keras.models.Sequential()

    # Input layer
    nn.add(tf.keras.layers.Dense(units=4, activation="relu", input_dim=features_count))

    # Second layer
    nn.add(tf.keras.layers.Dense(units=2, activation="relu"))

    # Third layer
    nn.add(tf.keras.layers.Dense(units=2, activation="relu"))

    # Output layer
    nn.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

    # Check the structure of the model
    nn.summary()

    # Compile the model
    nn.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])

    # Train the model
    fit_nn = nn.fit(X_train, y_train, epochs=500)

    # Evaluate the model using the test data
    model_loss, model_accuracy = nn.evaluate(X_test,y_test,verbose=2)
    print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")

Also mirrors that of Caleb Wolf's "AlphabetSoupCharity_Optimization.ipynb" file from the week 21 homework using the method provided in the starter file:

    The following was provided in those starter files:

        nn = tf.keras.models.Sequential()

    and:

        # Evaluate the model using the test data
        model_loss, model_accuracy = nn.evaluate(X_test_scaled,y_test,verbose=2)
        print(f"Loss: {model_loss}, Accuracy: {model_accuracy}")


    The following code:

        # Compile the model
        nn.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuaracy"])

    and:

        # Train the model
        it_nn = nn.fit(X_train_scaled, y_train, epochs=100)

    Were from W21_Class2_Activities 01-Ins-Over_the_Moon_NN.ipynb


I used the following method:

    import dill
    wd = "/whatever/you/want/your/working/directory/to/be/"
    rf= RandomForestRegressor(n_estimators=250, max_features=9,compute_importances=True)
    rf.fit(Predx, Predy)
    dill.dump(rf, open(wd + "filename.obj","wb"))

    model = dill.load(open(wd + "filename.obj","rb"))

From the following source:

    https://stackoverflow.com/questions/20662023/save-python-random-forest-model-to-file

To save our models:

    # Export the Trained Models
    import pickle

    with open ("/content/drive/MyDrive/Colab Notebooks/Building_Model_Exports/scaler_model","wb") as f:
    pickle.dump(features_scaler,f)

    with open ("/content/drive/MyDrive/Colab Notebooks/Building_Model_Exports/pca_model","wb") as f:
    pickle.dump(pca_model,f)

    with open ("/content/drive/MyDrive/Colab Notebooks/Building_Model_Exports/rf_model","wb") as f:
    pickle.dump(rf_model,f)

And to load them:

    # Test Importing Trained Models

    test_scaler = pickle.load(open("/content/drive/MyDrive/Colab Notebooks/Building_Model_Exports/scaler_model","rb"))

    test_pca = pickle.load(open("/content/drive/MyDrive/Colab Notebooks/Building_Model_Exports/pca_model","rb"))

    test_rf = pickle.load(open("/content/drive/MyDrive/Colab Notebooks/Building_Model_Exports/rf_model","rb"))


--------------------------------------------------
Creating_Application.ipynb
-------------------------------------------------

This Jupyter Notebook was used to export the distinct values in the wrangled data into CSV files for use in binding HTML dropdown values.

One example (weather conditions) can be seen here:

    # Create Dataframe for Weather
    weather_conditions_list = wrangled_df["Weather Condition"].unique()
    # Create Dataframe of Unique Values
    weather_conditions_df = pd.DataFrame(weather_conditions_list)
    # Rename Column
    weather_conditions_df=weather_conditions_df.rename(columns={0:"Weather"})
    # Sort Alphabetically
    weather_conditions_df = weather_conditions_df.sort_values("Weather", ascending=True)
    # Reset Index
    weather_conditions_df = weather_conditions_df.reset_index(drop=True)
    # Export New Dataframe
    weather_conditions_df.to_csv("Reference_Data/Weather_Data.csv")


--------------------------------------------------
hosting.py
-------------------------------------------------

The Flask application used to populate dropdown data as well as receive matchup data and perform the prediction, using the exported models from the Building Model phase.  Added a route for each dropdown as well as a POST route that received JSON data from the form element.

The starter code for the wizard HTML template used as the main web user interface was obtained here:

    https://www.w3schools.com/howto/howto_js_form_steps.asp?

