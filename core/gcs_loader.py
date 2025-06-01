# core/gcs_loader.py V33.GCS_LOADER_C - Remplacement import config
import logging
import os
from pathlib import Path
from google.cloud import storage
import shutil 

try:
    # --- MODIFICATION ICI ---
    import emergence_config
    # --- FIN MODIFICATION ---
except ImportError:
    if __name__ == '__main__' and __package__ is None:
        import sys
        sys.path.append(str(Path(__file__).resolve().parent.parent))
    # --- MODIFICATION ICI ---
    import emergence_config
    # --- FIN MODIFICATION ---

logger = logging.getLogger("gcs_loader")

def download_file_from_gcs(bucket_name: str, source_blob_name: str, destination_file_path: Path) -> bool:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    try:
        destination_file_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(destination_file_path))
        logger.info(f"Téléchargé GCS gs://{bucket_name}/{source_blob_name} vers {destination_file_path}")
        return True
    except Exception as e:
        logger.error(f"Erreur téléchargement GCS gs://{bucket_name}/{source_blob_name} vers {destination_file_path}: {e}", exc_info=True)
        return False

def download_directory_from_gcs(bucket_name: str, source_directory_prefix: str, destination_local_path: Path) -> bool:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=source_directory_prefix)) # Convertir en liste pour vérifier si vide
    
    # --- MODIFICATION ICI V33.GCS_LOADER_B -> V33.GCS_LOADER_C ---
    # Nettoyer le dossier de destination uniquement si des blobs existent réellement.
    if blobs:
        if destination_local_path.exists():
            try:
                shutil.rmtree(destination_local_path)
                logger.info(f"Ancien contenu de {destination_local_path} supprimé avant téléchargement GCS.")
            except Exception as e_rm:
                logger.error(f"Erreur suppression {destination_local_path}: {e_rm}")
        destination_local_path.mkdir(parents=True, exist_ok=True)
    elif not destination_local_path.exists(): # Si pas de blobs et dossier local n'existe pas, le créer vide.
        destination_local_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Aucun blob trouvé pour {source_directory_prefix}, dossier local {destination_local_path} créé vide (ou existait déjà vide).")
    else: # Pas de blobs, mais le dossier local existe, on ne le touche pas.
        logger.info(f"Aucun blob trouvé pour {source_directory_prefix}, dossier local {destination_local_path} conservé tel quel.")
    # --- FIN MODIFICATION ---

    downloaded_count = 0
    for blob in blobs: # Itérer sur la liste de blobs (peut être vide)
        if blob.name.endswith('/'): 
            continue
        relative_path = Path(blob.name).relative_to(Path(source_directory_prefix))
        local_file_path = destination_local_path / relative_path
        try:
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_file_path))
            logger.debug(f"Téléchargé GCS gs://{bucket_name}/{blob.name} vers {local_file_path}")
            downloaded_count +=1
        except Exception as e:
            logger.error(f"Erreur téléchargement GCS gs://{bucket_name}/{blob.name} vers {local_file_path}: {e}", exc_info=True)
    
    if downloaded_count > 0:
        logger.info(f"{downloaded_count} fichier(s) téléchargé(s) depuis GCS gs://{bucket_name}/{source_directory_prefix}/ vers {destination_local_path}")
    elif not blobs: # Si blobs était vide
        logger.info(f"Aucun fichier à télécharger depuis GCS gs://{bucket_name}/{source_directory_prefix}/ (source vide).")
    else: # Si blobs n'était pas vide, mais downloaded_count est 0 (erreurs de téléchargement)
        logger.warning(f"Tentative de téléchargement depuis GCS gs://{bucket_name}/{source_directory_prefix}/ mais aucun fichier n'a pu être téléchargé (erreurs?).")
    
    return downloaded_count > 0 or not blobs # Succès si des fichiers sont téléchargés OU s'il n'y avait rien à télécharger

def initialize_data_from_gcs(force_rebuild_db: bool = False) -> bool:
    # --- MODIFICATION ICI ---
    if hasattr(emergence_config, 'IS_GCP_ENVIRONMENT') and emergence_config.IS_GCP_ENVIRONMENT and \
       hasattr(emergence_config, 'GCS_BUCKET_NAME') and emergence_config.GCS_BUCKET_NAME:
        # --- FIN MODIFICATION ---
        logger.info("Initialisation des données depuis GCS en cours...")

        # --- MODIFICATION ICI ---
        gcs_local_data_dir = emergence_config.GCS_LOCAL_DATA_BASE_PATH 
        # --- FIN MODIFICATION ---
        gcs_local_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Répertoire local pour données GCS: {gcs_local_data_dir}")

        all_downloads_successful = True
        path_to_downloaded_json = None
        path_to_downloaded_memoire_txt = None
        dest_archives_dir = None

        # --- MODIFICATION ICI (pour tous les hasattr et accès à config) ---
        if hasattr(emergence_config, 'GCS_REMOTE_MEMORY_FILE_NAME') and emergence_config.GCS_REMOTE_MEMORY_FILE_NAME and \
           hasattr(emergence_config, 'LOCAL_MEMORY_FILE_NAME') and emergence_config.LOCAL_MEMORY_FILE_NAME:
            path_to_downloaded_memoire_txt = gcs_local_data_dir / emergence_config.LOCAL_MEMORY_FILE_NAME
            if not download_file_from_gcs(emergence_config.GCS_BUCKET_NAME, emergence_config.GCS_REMOTE_MEMORY_FILE_NAME, path_to_downloaded_memoire_txt):
                all_downloads_successful = False
        else:
            logger.warning("Config manquante pour GCS_REMOTE_MEMORY_FILE_NAME ou LOCAL_MEMORY_FILE_NAME. Skip memoire.txt.")

        if hasattr(emergence_config, 'GCS_REMOTE_PERSISTENCE_FILE_NAME') and emergence_config.GCS_REMOTE_PERSISTENCE_FILE_NAME and \
           hasattr(emergence_config, 'LOCAL_PERSISTENCE_FILE_NAME') and emergence_config.LOCAL_PERSISTENCE_FILE_NAME:
            path_to_downloaded_json = gcs_local_data_dir / emergence_config.LOCAL_PERSISTENCE_FILE_NAME
            if not download_file_from_gcs(emergence_config.GCS_BUCKET_NAME, emergence_config.GCS_REMOTE_PERSISTENCE_FILE_NAME, path_to_downloaded_json):
                all_downloads_successful = False
        else:
            logger.warning("Config manquante pour GCS_REMOTE_PERSISTENCE_FILE_NAME ou LOCAL_PERSISTENCE_FILE_NAME. Skip memoire_persistante.json.")

        if hasattr(emergence_config, 'GCS_REMOTE_ARCHIVES_DIR') and emergence_config.GCS_REMOTE_ARCHIVES_DIR:
            local_archives_sub_dir_name = getattr(emergence_config, 'LOCAL_ARCHIVES_DIR_NAME', Path(emergence_config.GCS_REMOTE_ARCHIVES_DIR.strip('/')).name)
            dest_archives_dir = gcs_local_data_dir / local_archives_sub_dir_name
            download_directory_from_gcs(emergence_config.GCS_BUCKET_NAME, emergence_config.GCS_REMOTE_ARCHIVES_DIR, dest_archives_dir)
            # La fonction download_directory_from_gcs retourne True si des fichiers sont téléchargés OU si le dossier source était vide.
            # Nous n'avons pas besoin de marquer all_downloads_successful = False ici, sauf si la fonction elle-même indiquait une erreur critique.
        else:
            logger.warning("Configuration manquante pour GCS_REMOTE_ARCHIVES_DIR. Skip archives.")
        # --- FIN MODIFICATION ---

        if not all_downloads_successful: # Concerne surtout memoire.txt et memoire_persistante.json
            logger.error("Au moins un téléchargement GCS critique (memoire.txt ou memoire_persistante.json) a échoué.")
            return False

        logger.info("Tentative de reconstruction de la base vectorielle ChromaDB après téléchargement GCS...")

        # --- MODIFICATION ICI ---
        chroma_db_final_path = Path(emergence_config.CHROMA_DB_PATH)
        # --- FIN MODIFICATION ---
        chroma_db_final_path.mkdir(parents=True, exist_ok=True)

        if force_rebuild_db and chroma_db_final_path.exists():
            logger.warning(f"FORCAGE REBUILD: Suppression du dossier DB vectorielle existant: {chroma_db_final_path}")
            try:
                shutil.rmtree(chroma_db_final_path)
                logger.info(f"Ancien dossier {chroma_db_final_path} supprimé pour reconstruction.")
                chroma_db_final_path.mkdir(parents=True, exist_ok=True)
            except OSError as e_rm_db:
                logger.error(f"Impossible de supprimer {chroma_db_final_path} pour reconstruction: {e_rm_db}")
                return False

        try:
            from build_vector_db import main as build_db_main

            logger.info(f"Appel de build_db_main avec json_path_arg='{str(path_to_downloaded_json) if path_to_downloaded_json else None}', "
                        # --- MODIFICATION ICI ---
                        f"vector_db_dir_arg='{emergence_config.CHROMA_DB_PATH}', "
                        # --- FIN MODIFICATION ---
                        f"archives_dir_override_arg='{str(dest_archives_dir) if dest_archives_dir else None}', "
                        f"memoire_txt_path_override_arg='{str(path_to_downloaded_memoire_txt) if path_to_downloaded_memoire_txt else None}'")

            build_db_main(
                json_path_arg=str(path_to_downloaded_json) if path_to_downloaded_json and path_to_downloaded_json.exists() else None, # S'assurer que le fichier existe
                # --- MODIFICATION ICI ---
                vector_db_dir_arg=emergence_config.CHROMA_DB_PATH,    
                chunk_size_arg=emergence_config.CHUNK_SIZE,
                chunk_overlap_arg=emergence_config.CHUNK_OVERLAP,
                # --- FIN MODIFICATION ---
                archives_dir_override_arg=str(dest_archives_dir) if dest_archives_dir and dest_archives_dir.exists() else None,
                memoire_txt_path_override_arg=str(path_to_downloaded_memoire_txt) if path_to_downloaded_memoire_txt and path_to_downloaded_memoire_txt.exists() else None,
                include_archives_arg=True, 
                include_memoire_txt_arg=True 
            )
            logger.info("Reconstruction de la base vectorielle ChromaDB terminée (ou tentée).")
            return True 

        except ImportError as e_import:
            logger.error(f"Impossible d'importer build_vector_db.main: {e_import}. La reconstruction de la DB doit être gérée autrement.", exc_info=True)
            return False
        except TypeError as e_type:
            logger.error(f"Erreur de type lors de l'appel à build_db_main: {e_type}", exc_info=True)
            return False
        except Exception as e_build:
            logger.error(f"Erreur lors de la tentative de reconstruction de ChromaDB: {e_build}", exc_info=True)
            return False
    else:
        logger.info("Pas d'environnement GCP ou GCS_BUCKET_NAME non défini. Initialisation GCS sautée.")
        # --- MODIFICATION ICI V33.GCS_LOADER_B : On s'attend à ce que build_vector_db soit appelé par app.py si besoin ---
        # Il n'est plus nécessaire de l'appeler ici en local car l'initialisation GCS est sautée.
        # Si on est en local, app.py doit gérer l'appel à build_vector_db.py directement avec les chemins locaux.
        # Cependant, pour la cohérence du flux d'initialisation, si on est PAS en GCP, mais que initialize_data_from_gcs
        # est quand même appelée (ce qui ne devrait pas arriver si app.py conditionne son appel à IS_GCP_ENVIRONMENT),
        # on pourrait vouloir lancer le build DB avec les chemins par défaut de config.
        # Pour l'instant, on retourne True et on laisse app.py gérer le build local.
        logger.info("Mode local : build_vector_db.py sera géré par app.py si nécessaire.")
        # --- FIN MODIFICATION ---
        return True

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - [%(name)s:%(lineno)d] %(message)s")
    if not os.getenv('GCS_BUCKET_NAME'):
        print("ERREUR: Variable d'environnement GCS_BUCKET_NAME non définie. Test annulé.")
    else:
        print(f"Test du chargement depuis GCS bucket: {os.getenv('GCS_BUCKET_NAME')}")
        # --- MODIFICATION ICI ---
        setattr(emergence_config, 'IS_GCP_ENVIRONMENT', True) 
        if not hasattr(emergence_config, 'GCS_LOCAL_DATA_BASE_PATH'):
            setattr(emergence_config, 'GCS_LOCAL_DATA_BASE_PATH', Path("./tmp_gcs_data_test_loader"))
        # S'assurer que les autres attributs de config sont là pour le test
        if not hasattr(emergence_config, 'GCS_REMOTE_MEMORY_FILE_NAME'): setattr(emergence_config, 'GCS_REMOTE_MEMORY_FILE_NAME', "data/memoire.txt")
        if not hasattr(emergence_config, 'LOCAL_MEMORY_FILE_NAME'): setattr(emergence_config, 'LOCAL_MEMORY_FILE_NAME', "memoire.txt")
        if not hasattr(emergence_config, 'GCS_REMOTE_PERSISTENCE_FILE_NAME'): setattr(emergence_config, 'GCS_REMOTE_PERSISTENCE_FILE_NAME', "data/memoire_persistante.json")
        if not hasattr(emergence_config, 'LOCAL_PERSISTENCE_FILE_NAME'): setattr(emergence_config, 'LOCAL_PERSISTENCE_FILE_NAME', "memoire_persistante.json")
        if not hasattr(emergence_config, 'GCS_REMOTE_ARCHIVES_DIR'): setattr(emergence_config, 'GCS_REMOTE_ARCHIVES_DIR', "data/archives/")
        if not hasattr(emergence_config, 'CHROMA_DB_PATH'): setattr(emergence_config, 'CHROMA_DB_PATH', "./tmp_chroma_db_test_loader")
        if not hasattr(emergence_config, 'CHUNK_SIZE'): setattr(emergence_config, 'CHUNK_SIZE', 1000)
        if not hasattr(emergence_config, 'CHUNK_OVERLAP'): setattr(emergence_config, 'CHUNK_OVERLAP', 150)
        # --- FIN MODIFICATION ---
        success = initialize_data_from_gcs(force_rebuild_db=True)
        print(f"Test de chargement terminé. Succès: {success}. Vérifiez les logs et le contenu.")