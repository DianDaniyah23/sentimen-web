/**
 * Fungsi untuk membuat Pie Chart tunggal.
 * @param {string} canvasId - ID dari elemen <canvas>
 * @param {string} chartLabel - Judul chart
 * @param {object} data - Objek berisi data, contoh: { Positif: 50, Negatif: 20, Netral: 30 }
 */
function createPieChart(canvasId, chartLabel, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    
    // Hancurkan chart lama jika ada untuk mencegah kedipan aneh saat re-render
    if (window.myCharts && window.myCharts[canvasId]) {
        window.myCharts[canvasId].destroy();
    }
    
    // Pastikan window.myCharts ada
    if (!window.myCharts) {
        window.myCharts = {};
    }

    // Buat chart baru
    window.myCharts[canvasId] = new Chart(ctx, {
        type: 'pie',
        data: {
            labels: ['Positif', 'Negatif', 'Netral'],
            datasets: [{
                label: chartLabel,
                data: [data.Positif, data.Negatif, data.Netral],
                backgroundColor: [
                    'rgba(25, 135, 84, 0.8)',  // Hijau
                    'rgba(220, 53, 69, 0.8)',   // Merah
                    'rgba(108, 117, 125, 0.8)'  // Abu-abu
                ],
                borderColor: [
                    'rgba(25, 135, 84, 1)',
                    'rgba(220, 53, 69, 1)',
                    'rgba(108, 117, 125, 1)'
                ],
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: chartLabel,
                    font: {
                        size: 16
                    }
                }
            }
        }
    });
}

/**
 * Fungsi untuk membuat Bar Chart perbandingan.
 * @param {string} canvasId - ID dari elemen <canvas>
 * @param {object} allData - Objek berisi data dari semua metode
 */
function createBarChart(canvasId, allData) {
    const ctx = document.getElementById(canvasId).getContext('2d');
    const labels = ['Positif', 'Negatif', 'Netral'];
    
    // Hancurkan chart lama jika ada
    if (window.myCharts && window.myCharts[canvasId]) {
        window.myCharts[canvasId].destroy();
    }
     // Pastikan window.myCharts ada
    if (!window.myCharts) {
        window.myCharts = {};
    }

    window.myCharts[canvasId] = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'TextBlob',
                    data: [allData.textblob.Positif, allData.textblob.Negatif, allData.textblob.Netral],
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: 'VADER',
                    data: [allData.vader.Positif, allData.vader.Negatif, allData.vader.Netral],
                    backgroundColor: 'rgba(255, 99, 132, 0.7)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Naive Bayes',
                    data: [allData.naive_bayes.Positif, allData.naive_bayes.Negatif, allData.naive_bayes.Netral],
                    backgroundColor: 'rgba(75, 192, 192, 0.7)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Perbandingan Hasil Analisis Sentimen (Jumlah Komentar)',
                    font: {
                        size: 18
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        stepSize: 1 // Pastikan skala y adalah bilangan bulat
                    }
                }
            }
        }
    });
}
