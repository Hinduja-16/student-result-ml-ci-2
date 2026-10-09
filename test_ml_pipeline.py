def test_high_performance_student(self):
        model = joblib.load("student_result_model.pkl")

        sample = pd.DataFrame([{
            "attendance": 90,
            "internal_marks": 85,
            "assignment_marks": 88,
            "previous_score": 80
        }])

        prediction = model.predict(sample)[0]
        self.assertEqual(int(prediction), 1)
