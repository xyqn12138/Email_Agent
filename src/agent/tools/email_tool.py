from langchain_core.tools import tool

@tool("send_email", return_direct=True)
def send_email(to: str, subject: str, body:str) -> str:
    """
    发送电子邮件工具函数。

    Args:
        to: 收件人的邮箱地址
        subject : 邮件的主题
        body : 邮件的正文内容

    Returns:
        str: A confirmation message indicating that the email was sent.
    """
    # 这里可以添加实际的发送邮件逻辑，例如使用SMTP库或第三方邮件服务API。
    return f"Email sent to {to} with subject '{subject}' and body '{body}'"

