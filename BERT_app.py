import os
import sys
from colorama import init, Fore, Style
from manga_recommendation_content_filter import get_recommendations_from_prompt

# Initialiser colorama
init()


def clear_console():
    os.system('cls' if os.name == 'nt' else 'clear')


def main():
    clear_console()
    print(Fore.CYAN + "Bienvenue dans le système de recommandation de mangas!" + Style.RESET_ALL)
    print("Entrez un titre, une phrase ou un genre pour obtenir des recommandations.")
    print("Tapez 'exit' pour quitter l'application.")

    while True:
        prompt = input(Fore.YELLOW + "Votre prompt: " + Style.RESET_ALL)
        if prompt.lower() == 'exit':
            print(
                Fore.GREEN + "Merci d'avoir utilisé le système de recommandation de mangas. Au revoir!" + Style.RESET_ALL)
            break

        recommendations = get_recommendations_from_prompt(prompt)
        print(Fore.CYAN + f"Recommandations pour '{prompt}':" + Style.RESET_ALL)
        for idx, title in enumerate(recommendations, 1):
            print(Fore.BLUE + f"{idx}. {title}" + Style.RESET_ALL)
        print("\n")


if __name__ == "__main__":
    main()
