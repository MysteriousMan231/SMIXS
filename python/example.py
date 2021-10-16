from src.SMIXSHelper   import init_penatly_matrix
from src.DataGenerator import generate_dataset

def main():
    
    #init_penatly_matrix(10)

    generate_dataset(number_of_clusters = 5)


if __name__ == "__main__":
    main()