import streamlit as st
def intro():

    st.write("# Chào mừng đến với chương trình dự đoán giá chứng khoán! 👋")
    st.sidebar.success("Chọn một lựa chọn")

    st.markdown(
        """
        Đây là ứng dụng dùng để kiểm tra độ hiệu quả các mô hình machine learning được train từ trước

        👈 Hãy lựa chọn sử dụng một mô hình cụ thể để dự đoán hoặc so sánh kết quả dự đoán của tất cả các mô hình
        """
    )