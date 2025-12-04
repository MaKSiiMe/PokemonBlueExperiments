import os
import glob
import subprocess
import sys

def main():
    # Liste tous les .state
    state_files = glob.glob("*.state")
    
    if not state_files:
        print("Aucun fichier .state trouvé.")
        return

    print(f"🔥 Démarrage de la PURGE sur {len(state_files)} fichiers...")
    print("Les fichiers corrompus seront supprimés définitivement.\n")

    good_count = 0
    bad_count = 0

    for state in state_files:
        print(f"Testing {state}...", end=" ")
        
        # On lance le test dans une bulle isolée
        # Si ça crash, ça n'arrête pas ce script
        try:
            # On utilise le même python que celui en cours
            result = subprocess.run(
                [sys.executable, "src/utils/check_single_state.py", state],
                capture_output=True,
                timeout=5 # Si ça freeze plus de 5s, on considère que c'est mort
            )

            if result.returncode == 0:
                print("✅ OK")
                good_count += 1
            else:
                print("❌ CRASH -> SUPPRESSION")
                os.remove(state)
                bad_count += 1

        except subprocess.TimeoutExpired:
            print("⏰ TIMEOUT (Freeze) -> SUPPRESSION")
            os.remove(state)
            bad_count += 1
        except Exception as e:
            print(f"⚠️ Erreur script: {e}")

    print("-" * 30)
    print(f"🏁 Terminé.")
    print(f"Survivants : {good_count}")
    print(f"Éliminés   : {bad_count}")

if __name__ == "__main__":
    main()
