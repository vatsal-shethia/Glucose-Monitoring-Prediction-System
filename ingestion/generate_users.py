import pandas as pd
import numpy as np

np.random.seed(42)

def generate_users(num_users=8):
    """
    Generate basic user data
    """

    user_ids = np.arange(1, num_users + 1)

    ages = np.random.randint(20, 60, size=num_users)

    genders = np.random.choice(["Male", "Female"], size=num_users)

    users_df = pd.DataFrame({
        "user_id": user_ids,
        "age": ages,
        "gender": genders
    })

    return users_df


def save_users(users_df):
    """
    Save users to CSV
    """
    users_df.to_csv("data/users.csv", index=False)
    print("✅ users.csv created")


if __name__ == "__main__":
    users = generate_users(num_users=8)
    save_users(users)