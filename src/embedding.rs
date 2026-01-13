//! Embedded MiniLM-v2 model for semantic similarity
//!
//! Provides text embedding using a statically-linked ONNX model.
//! Used for convergence detection in introspection loops.

use ort::session::Session;
use ort::value::Tensor;
use std::sync::Mutex;

// Static model bytes - ~23MB each, embedded at compile time
#[cfg(target_arch = "x86_64")]
static MODEL_ONNX: &[u8] = include_bytes!("../data/embedding/model_quint8_avx2.onnx");

#[cfg(target_arch = "aarch64")]
static MODEL_ONNX: &[u8] = include_bytes!("../data/embedding/model_qint8_arm64.onnx");

// Fallback for other architectures (use x86 model, may be slower)
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
static MODEL_ONNX: &[u8] = include_bytes!("../data/embedding/model_quint8_avx2.onnx");

static TOKENIZER_JSON: &[u8] = include_bytes!("../data/embedding/tokenizer.json");

static SESSION: Mutex<Option<Session>> = Mutex::new(None);
static TOKENIZER: Mutex<Option<tokenizers::Tokenizer>> = Mutex::new(None);

/// Initialize the embedding model (lazy, thread-safe)
fn with_session<T>(
    f: impl FnOnce(&mut Session) -> Result<T, EmbeddingError>,
) -> Result<T, EmbeddingError> {
    let mut guard = SESSION.lock().map_err(|_| EmbeddingError::Lock)?;
    if guard.is_none() {
        let session = Session::builder()?.commit_from_memory(MODEL_ONNX)?;
        *guard = Some(session);
    }
    f(guard.as_mut().unwrap())
}

fn with_tokenizer<T>(
    f: impl FnOnce(&tokenizers::Tokenizer) -> Result<T, EmbeddingError>,
) -> Result<T, EmbeddingError> {
    let mut guard = TOKENIZER.lock().map_err(|_| EmbeddingError::Lock)?;
    if guard.is_none() {
        let tokenizer = tokenizers::Tokenizer::from_bytes(TOKENIZER_JSON)
            .map_err(|e| EmbeddingError::Tokenizer(e.to_string()))?;
        *guard = Some(tokenizer);
    }
    f(guard.as_ref().unwrap())
}

/// Generate an embedding vector for the given text
pub fn embed(text: &str) -> Result<Vec<f32>, EmbeddingError> {
    // Tokenize first (separate lock)
    let (input_ids, attention_mask, token_type_ids, seq_len) = with_tokenizer(|tokenizer| {
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| EmbeddingError::Tokenizer(e.to_string()))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();
        let token_type_ids: Vec<i64> = vec![0i64; input_ids.len()];
        let seq_len = input_ids.len();

        Ok((input_ids, attention_mask, token_type_ids, seq_len))
    })?;

    // Run inference
    with_session(|session| {
        // Create input tensors using ort's tuple format (shape, data)
        let input_ids_tensor = Tensor::from_array(([1, seq_len], input_ids))?;
        let attention_mask_tensor = Tensor::from_array(([1, seq_len], attention_mask))?;
        let token_type_ids_tensor = Tensor::from_array(([1, seq_len], token_type_ids))?;

        let outputs = session.run(ort::inputs![
            "input_ids" => input_ids_tensor,
            "attention_mask" => attention_mask_tensor,
            "token_type_ids" => token_type_ids_tensor,
        ])?;

        // Extract embeddings (mean pooling over token dimension)
        let output = &outputs["last_hidden_state"];
        let (shape, data) = output.try_extract_tensor::<f32>()?;

        // Shape is [1, seq_len, hidden_size]
        let hidden_size = shape[2] as usize;

        // Mean pooling: average across sequence length dimension
        let mut embedding = vec![0.0f32; hidden_size];
        for i in 0..seq_len {
            for j in 0..hidden_size {
                embedding[j] += data[i * hidden_size + j];
            }
        }
        for v in &mut embedding {
            *v /= seq_len as f32;
        }

        // L2 normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut embedding {
                *v /= norm;
            }
        }

        Ok(embedding)
    })
}

/// Compute cosine similarity between two texts
pub fn similarity(text_a: &str, text_b: &str) -> Result<f32, EmbeddingError> {
    let emb_a = embed(text_a)?;
    let emb_b = embed(text_b)?;

    // Cosine similarity (embeddings are already normalized)
    let dot: f32 = emb_a.iter().zip(&emb_b).map(|(a, b)| a * b).sum();
    Ok(dot)
}

/// Check if two texts have converged (similarity above threshold)
pub fn has_converged(text_a: &str, text_b: &str, threshold: f32) -> Result<bool, EmbeddingError> {
    let sim = similarity(text_a, text_b)?;
    Ok(sim >= threshold)
}

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("ONNX runtime error: {0}")]
    Ort(#[from] ort::Error),

    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Lock poisoned")]
    Lock,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embed_produces_vector() {
        let embedding = embed("This is a test sentence.").unwrap();
        assert_eq!(embedding.len(), 384); // MiniLM-v2 produces 384-dim vectors
    }

    #[test]
    fn test_similarity_same_text() {
        let sim = similarity("hello world", "hello world").unwrap();
        assert!(
            sim > 0.99,
            "Same text should have similarity ~1.0, got {}",
            sim
        );
    }

    #[test]
    fn test_similarity_different_texts() {
        // Test that unrelated texts have LOWER similarity than related texts
        let unrelated = similarity(
            "The cat sat on the mat",
            "Quantum physics explains subatomic behavior",
        )
        .unwrap();
        let related =
            similarity("The cat sat on the mat", "A feline was sitting on a rug").unwrap();
        assert!(
            related > unrelated,
            "Related texts ({}) should have higher similarity than unrelated ({})",
            related,
            unrelated
        );
    }

    #[test]
    fn test_similarity_semantically_similar() {
        let sim = similarity(
            "The dog runs in the park",
            "A canine is running through a garden",
        )
        .unwrap();
        assert!(
            sim > 0.5,
            "Semantically similar texts should have high similarity, got {}",
            sim
        );
    }

    #[test]
    fn test_convergence_detection() {
        let converged = has_converged(
            "This module handles user authentication",
            "This module manages user auth and login",
            0.7,
        )
        .unwrap();
        assert!(converged, "Similar descriptions should converge");

        let not_converged = has_converged(
            "This module handles user authentication",
            "This module parses JSON data",
            0.9,
        )
        .unwrap();
        assert!(
            !not_converged,
            "Different descriptions should not converge at high threshold"
        );
    }
}
