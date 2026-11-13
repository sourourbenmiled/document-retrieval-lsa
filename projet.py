import numpy as np
import matplotlib.pyplot as plt
import time

### ==== 1. SIMILARITÉ COSINUS ==== ###
def cosine_similarity(q, d):
    dot_product = np.dot(q, d)
    norm_q = np.linalg.norm(q)
    norm_d = np.linalg.norm(d)
    if norm_q == 0 or norm_d == 0:
        return 0
    return dot_product / (norm_q * norm_d)

### ==== 2. DÉCOMPOSITION SVD MANUELLE ==== ###
def manual_svd(D, k):
    k = min(k, min(D.shape))  # Ensure k doesn't exceed matrix rank
    AtA = D.T @ D
    eigvals_V, V = np.linalg.eigh(AtA)

    idx = np.argsort(eigvals_V)[::-1]
    eigvals_V = eigvals_V[idx]
    V = V[:, idx]

    S = np.sqrt(np.maximum(eigvals_V[:k], 0))  # Handle negative eigenvalues
    Sk = np.diag(S)

    Uk = []
    for i in range(k):
        if S[i] > 1e-10:
            u_i = D @ V[:, i] / S[i]
            Uk.append(u_i / np.linalg.norm(u_i))  # Normalize Uk columns
        else:
            Uk.append(np.zeros(D.shape[0]))
    Uk = np.column_stack(Uk)

    return Uk, Sk, V[:, :k]

### ==== 3. BIDIAGONALIZATION ==== ###

def bidiagonalize(D):
    m, n = D.shape
    k = min(m, n)
    U = np.eye(m, dtype=np.float64)
    V = np.eye(n, dtype=np.float64)
    B = D.astype(np.float64).copy()

    for i in range(k):
        if i < m - 1:
            x = B[i:, i]
            sigma = np.sign(x[0]) * np.linalg.norm(x) if x[0] != 0 else np.linalg.norm(x)
            v = x.copy()
            v[0] += sigma
            v /= np.linalg.norm(v)
            # Apply Householder transformation directly from the left: B <- H @ B
            # H = I - 2 v v^T
            # So H B = B - 2 v (v^T B)
            B[i:, i:] -= 2.0 * np.outer(v, np.dot(v, B[i:, i:]))
            # Update U similarly: U <- U H^T = U H since H is symmetric
            U[:, i:] -= 2.0 * np.outer(U[:, i:] @ v, v)

        if i < n - 1 and i < m:
            x = B[i, i+1:]
            sigma = np.sign(x[0]) * np.linalg.norm(x) if x[0] != 0 else np.linalg.norm(x)
            v = x.copy()
            v[0] += sigma
            v /= np.linalg.norm(v)
            # Apply Householder transformation from the right: B <- B H
            # H = I - 2 v v^T
            # So B H = B - 2 (B v) v^T
            B[i:, i+1:] -= 2.0 * np.outer(np.dot(B[i:, i+1:], v), v)
            # Update V similarly: V <- V H
            V[:, i+1:] -= 2.0 * np.outer(V[:, i+1:] @ v, v)

    return U, B[:k, :k], V



### ==== 4. QR PAR ROTATION DE GIVENS ==== ###
def givens_rotation(a, b):
    if b == 0:
        return 1, 0
    r = np.hypot(a, b)
    return a / r, -b / r

def qr_givens(B):
    m, n = B.shape
    Q = np.eye(m, dtype=np.float64)
    R = B.copy()

    for j in range(n):
        for i in range(m - 1, j, -1):
            a, b = R[i - 1, j], R[i, j]
            if abs(b) > 1e-10:
                c, s = givens_rotation(a, b)
                # Apply Givens rotation directly to R and Q
                R[[i - 1, i], j] = [c * a - s * b, s * a + c * b]
                R[i, j] = 0  # Set the below diagonal element to zero
                Q[:, [i - 1, i]] = Q[:, [i - 1, i]] @ np.array([[c, -s], [s, c]])

    return Q, R


### ==== 5. RECONSTRUCTION ERROR ==== ###
def reconstruction_error(D, k):
    Uk, Sk, Vk = manual_svd(D, k)
    Dk = Uk @ Sk @ Vk.T
    return np.linalg.norm(D - Dk)

### ==== 6. CALCUL DES SCORES ==== ###
def compute_scores(D, q, method="basic", k=None, max_iter=100, tol=1e-6):
    Nd = D.shape[1]

    if method == "basic":
        return np.array([cosine_similarity(q, D[:, j]) for j in range(Nd)])

    elif method == "svd":
        if k is None:
            k = min(D.shape)
        Uk, Sk, Vk = manual_svd(D, k)
        q_proj = Uk.T @ q
        q_proj = q_proj / np.linalg.norm(q_proj) if np.linalg.norm(q_proj) > 0 else q_proj
        return np.array([cosine_similarity(q_proj, Sk @ Vk.T[:, j]) for j in range(Nd)])

    elif method == "bidiag_qr":
        if k is None:
            k = min(D.shape)
        m, n = D.shape

        # Bidiagonalization
        U, B, V = bidiagonalize(D)
        B_k = B[:k, :k].copy()
        V_k = np.eye(k, dtype=np.float64)

        # QR iterations
        for _ in range(max_iter):
            Q, R = qr_givens(B_k)
            B_k = R @ Q
            V_k = V_k @ Q
            off_diag = np.abs(np.diag(B_k, k=1))
            if np.all(off_diag < tol):
                break

        # Get singular values and vectors
        S_approx = np.abs(np.diag(B_k))
        idx = np.argsort(S_approx)[::-1][:k]  # Sort descending
        S_approx = S_approx[idx]
        
        # Align matrices with sorted singular values
        Uk = U[:, :k] @ Q[:, idx]
        Vk_k = V[:, :k] @ V_k[:, idx]

        # Project query
        q_proj = Uk.T @ q
        q_proj = q_proj / np.linalg.norm(q_proj) if np.linalg.norm(q_proj) > 0 else q_proj

        # Compute scores
        scores = []
        for j in range(Nd):
            # Project document using Uk (not Vk_k)
            d_proj = Uk.T @ D[:, j]
            # Apply singular values
            d_scaled = S_approx * d_proj  # Element-wise multiplication
            score = cosine_similarity(q_proj, d_scaled)
            scores.append(score)

        return np.array(scores)

    
### ==== 5. EXEMPLES ==== ###
def exemple_1():
    doc_names = ["D₁: « Algèbre linéaire et matrices »", 
                 "D₂: « Analyse réelle et suites »", 
                 "D₃: « Probabilités et statistiques »", 
                 "D₄: « Matrices et déterminants »"]
    
    D = np.array([
        [1, 0, 0, 0],  # algèbre
        [0, 1, 0, 0],  # analyse
        [0, 0, 0, 1],  # déterminants
        [1, 0, 0, 1],  # matrices
        [0, 0, 1, 0],  # probabilités
        [0, 0, 1, 0],  # statistiques
        [0, 1, 0, 0]   # suites
    ])

    q = np.array([1, 0, 0, 1, 0, 0, 0])

    print("=== Exemple 1 ===")

    print(" \nRequête: matrices, algèbre\n")

    scores_basic = compute_scores(D, q, method="basic")

    print("Scores (Basic):")
    for i, score in enumerate(scores_basic):
        print(f"  {doc_names[i]}: {score:.4f}")

    scores_svd = compute_scores(D, q, method="svd", k=2)
    print("\nScores (SVD, k=2):")
    for i, score in enumerate(scores_svd):
        print(f"  {doc_names[i]}: {score:.4f}")

    scores_bidiag = compute_scores(D, q, method="bidiag_qr", k=2)
    print("\nScores (Bidiag+QR, k=2):")
    for i, score in enumerate(scores_bidiag):
        print(f"  {doc_names[i]}: {score:.4f}")

def exemple_3():
    # Noms des documents
    doc_names = [
        "Doc1: Croissance, PIB, Investissement",
        "Doc2: Inflation, Monnaie, Dépression", 
        "Doc3: Commerce, Exportation, Croissance",
        "Doc4: Emploi, Chômage, Salaires",
        "Doc5: Impôts, Fiscalité, Revenu",
        "Doc6: Géologie, Faille, Tremblement",
        "Doc7: Volcan, Séisme, Plaque tectonique",
        "Doc8: Dépression, Bassin, Érosion",
        "Doc9: Stratigraphie, Couches, Roche",
        "Doc10: Gisement, Forage, Bassin"
    ]
    
    # Liste des termes uniques triés par ordre alphabétique
    vocab = sorted({
        "croissance", "PIB", "investissement",
        "inflation", "monnaie", "dépression",
        "commerce", "exportation", "croissance",
        "emploi", "chômage", "salaires",
        "impôts", "fiscalité", "revenu",
        "géologie", "faille", "tremblement",
        "volcan", "séisme", "plaque tectonique",
        "dépression", "bassin", "érosion",
        "stratigraphie", "couches", "roche",
        "gisement", "forage", "bassin"
    })
    term_to_index = {term: i for i, term in enumerate(vocab)}

    # Construction de la matrice D (termes × documents)
    D = np.zeros((len(vocab), 10))
    docs = [
        ["croissance", "PIB", "investissement"],
        ["inflation", "monnaie", "dépression"],
        ["commerce", "exportation", "croissance"],
        ["emploi", "chômage", "salaires"],
        ["impôts", "fiscalité", "revenu"],
        ["géologie", "faille", "tremblement"],
        ["volcan", "séisme", "plaque tectonique"],
        ["dépression", "bassin", "érosion"],
        ["stratigraphie", "couches", "roche"],
        ["gisement", "forage", "bassin"]
    ]
    for j, doc in enumerate(docs):
        for term in doc:
            D[term_to_index[term], j] = 1

    # Requêtes
    q2 = np.zeros(len(vocab))
    for term in ["dépression", "croissance"]:
        q2[term_to_index[term]] = 1

    q3 = np.zeros(len(vocab))
    for term in ["bassin", "fiscalité"]:
        q3[term_to_index[term]] = 1

    print("=== Exemple 3 ===")
    
    # (a) Classement par score pour différents k
    print("\n(a) Classement des documents par score de pertinence")
    for k in [2, 4, 6, 8]:
        print(f"\nRang k = {k}")
        
        # Requête q2
        scores_q2 = compute_scores(D, q2, method="svd", k=k)
        print(f"q2 (dépression, croissance):")
        for i in np.argsort(scores_q2)[::-1]:  # Tri décroissant
            print(f"  {doc_names[i]}: {scores_q2[i]:.4f}")
        
        # Requête q3
        scores_q3 = compute_scores(D, q3, method="svd", k=k)
        print(f"\nq3 (bassin, fiscalité):")
        for i in np.argsort(scores_q3)[::-1]:  # Tri décroissant
            print(f"  {doc_names[i]}: {scores_q3[i]:.4f}")

    # (b) Erreur de reconstruction
    print("\n(b) Erreur de reconstruction ||D - Dk||₂")
    errors = []
    max_k = min(D.shape)
    for k in range(1, max_k + 1):
        Uk, Sk, Vk = manual_svd(D, k)
        Dk = Uk @ Sk @ Vk.T
        error = np.linalg.norm(D - Dk)
        errors.append(error)
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, max_k + 1), errors, marker='o')
    plt.xlabel("Rang k")
    plt.ylabel("Erreur de reconstruction ||D - Dk||₂")
    plt.title("Evolution de l'erreur de reconstruction en fonction de k")
    plt.grid(True)

    # (c) Analyse de la polysémie avec différents seuils
    print("\n(c) Effet de la polysémie avec différents seuils")
    
    k = 8  # Rang fixé pour cette analyse
    scores_q2 = compute_scores(D, q2, method="svd", k=k)
    scores_q3 = compute_scores(D, q3, method="svd", k=k)
    
    seuil= 0.5
    print(f"\nSeuil = 0.5")
                                                                        
    # Documents pertinents pour q2
    print(f"q2 (dépression,croissance):")
    for i, score in enumerate(scores_q2):
        if score > seuil:
            print(f"  {doc_names[i]}: {score:.4f}")
                                                                        
    # Documents pertinents pour q3
    print(f"\nq3 (bassin,fiscalité):")
    for i, score in enumerate(scores_q3):
        if score > seuil:
            print(f"  {doc_names[i]}: {score:.4f}")
                                                                        
    # Analyse de la polysémie pour "dépression"
    doc2_pertinent = scores_q2[1] > seuil  # Doc2
    doc8_pertinent = scores_q2[7] > seuil  # Doc8
    if doc2_pertinent and doc8_pertinent:
        print("\n→ Polysémie détectée: 'dépression' apparaît dans des contextes différents")
        print(f"   - {doc_names[1]}")
        print(f"   - {doc_names[7]}")

### ==== 7. benchmark ==== ###

def benchmark():
    # Configuration du benchmark
    Nd_values = [5, 10, 15, 20, 50, 100, 200]
    k_fixed = 5  # Valeur fixe pour k dans SVD et Bidiag+QR
    n_repeats = 3  # Nombre de répétitions pour lissage des mesures
    methods = ['basic', 'svd', 'bidiag_qr']
    colors = {'basic': 'blue', 'svd': 'green', 'bidiag_qr': 'red'}
    
    # Stockage des résultats
    results = {method: np.zeros(len(Nd_values)) for method in methods}
    
    # Warm-up pour éviter les biais de démarrage
    D_warmup = np.random.randint(0, 2, (30, 10))
    q_warmup = np.random.randint(0, 2, 30)
    for _ in range(3):
        for method in methods:
            compute_scores(D_warmup, q_warmup, method=method, k=k_fixed)

    # Boucle principale de benchmark
    for i, Nd in enumerate(Nd_values):
        Nt = 3 * Nd
        print(f"\nBenchmark en cours - Nd={Nd}, Nt={Nt}...")
        
        for method in methods:
            times = []
            for _ in range(n_repeats):
                # Génération de données aléatoires
                D = np.random.randint(0, 2, (Nt, Nd))
                q = np.random.randint(0, 2, Nt)
                
                # Mesure du temps
                start = time.perf_counter()
                compute_scores(D, q, method=method, k=k_fixed if method != 'basic' else None)
                times.append(time.perf_counter() - start)
            
            # Moyenne des temps (en excluant le premier résultat)
            results[method][i] = np.mean(times[1:]) if n_repeats > 1 else times[0]
    
    # Affichage des résultats
    plt.figure(figsize=(12, 6))
    for method in methods:
        plt.plot(Nd_values, results[method], 
                marker='o', 
                linestyle='-', 
                color=colors[method], 
                label=f"{method.upper()}{' (k=5)' if method != 'basic' else ''}")
    
    plt.xlabel("Nombre de documents (Nd)", fontsize=12)
    plt.ylabel("Temps d'exécution moyen (s)", fontsize=12)
    plt.title("Comparaison des temps de calcul des méthodes\n(Nt = 3×Nd, moyenne sur {} essais)".format(n_repeats), 
             fontsize=14, pad=20)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(Nd_values, fontsize=10)
    plt.yticks(fontsize=10)
    
    # Sauvegarde du graphique
    plt.tight_layout()
    plt.savefig('benchmark_results.png', dpi=300)
    plt.show()
    
    return results

### ==== 7. TRAITEMENT DE FICHIER TXT ==== ###
def traitement_documents_txt(path="documents.txt"):
    # Lecture du fichier avec gestion d'erreur
    try:
        with open(path, "r", encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"Erreur de lecture: {e}")
        return None

    # Extraction des données
    doc_data = []
    terms = set()
    
    for line in lines:
        parts = line.split()
        if len(parts) > 1:  # Doit avoir au moins un nom et un terme
            doc_name = parts[0]
            doc_terms = parts[1:]
            doc_data.append((doc_name, doc_terms))
            terms.update(doc_terms)
    
    if not doc_data:
        print("Aucun document valide trouvé")
        return None

    terms = sorted(terms)
    Nt, Nd = len(terms), len(doc_data)
    
    # Construction de la matrice
    D = np.zeros((Nt, Nd), dtype=int)
    term_to_idx = {term: i for i, term in enumerate(terms)}

    for j, (_, doc_terms) in enumerate(doc_data):
        for term in doc_terms:
            if term in term_to_idx:
                D[term_to_idx[term], j] = 1

    # Sélection aléatoire d'un terme de requête
    query_term = np.random.choice(terms)
    print(f"\nRequête choisie aléatoirement : {query_term}\n")
    
    # Vecteur de requête
    q = np.array([1 if term == query_term else 0 for term in terms])

    # Calcul des scores avec vérification des dimensions
    try:
        scores_basic = compute_scores(D, q, method="basic")
        k = min(10, Nt, Nd)
        print(" \nk= ",k,"\n")
        scores_svd = compute_scores(D, q, method="svd", k=k)
        scores_qrsvd = compute_scores(D, q, method="bidiag_qr", k=k)
        
        # Vérification que tous les scores ont la même dimension
        if len(scores_basic) != Nd or len(scores_svd) != Nd or len(scores_qrsvd) != Nd:
            print("Erreur: Dimensions des scores incompatibles")
            return None
            
    except Exception as e:
        print(f"Erreur dans le calcul des scores: {e}")
        return None

    # Affichage du tableau
    print(f"| {'Document':<40} | {'Sans SVD':<8} | {'Avec SVD':<8} | {'Avec Bidiag + QR':<8} |")
    print("|"+ "-"*42 + "|" + "-"*10 + "|" + "-"*10 + "|" + "-"*18 + "|")
    
    # Afficher seulement les documents avec score basic > 0
    for i in range(Nd):
        if scores_basic[i] > 0:
            doc_name = doc_data[i][0] + " " + " ".join(doc_data[i][1][:4])  # Limite à 4 termes
            print(f"| {doc_name:<40} | {scores_basic[i]:<8.4f} | {scores_svd[i]:<8.4f} | {scores_qrsvd[i]:<8.4f} |")
    
    return D 


### ==== 8. LANCEMENT GLOBAL ==== ###
if __name__ == "__main__":
    exemple_1()
    exemple_3()
    benchmark()
    traitement_documents_txt()
    plt.show() 