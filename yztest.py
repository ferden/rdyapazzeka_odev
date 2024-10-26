from ultralytics import YOLO
import matplotlib.pyplot as plt

# Modeli yükleme
model = YOLO('yolov8m.pt')  # YOLOv8 modelini kullanıyoruz

# Fotoğraflar ve video için çıkarım yapma, %50 confidence threshold ile
results_foto = model.predict(source='D:/yzeka/vscodefiles/fotograf', save=True)  # Fotoğraf klasörü yolu
results_video = model.predict(source='D:/yzeka/vscodefiles/video', save=True)  # Video dosyası yolu

# Tespit edilen nesnelerin sıklığını toplamak için bir liste oluşturun
detected_objects = []
for result in results_foto:  # Tüm sonuçlar için döngü
    detected_objects.extend([model.names[int(det.cls)] for det in result.boxes if det.conf >= 0.5])

print("Tespit Edilen Nesneler:", detected_objects)  # Tespit edilen nesneleri kontrol ediyoruz

# Eğer tespit edilen nesne yoksa uyarı verelim
if not detected_objects:
    print("Hiçbir nesne tespit edilmedi.")
else:
    # Nesnelerin sayısını ve başarı oranlarını hesaplayın
    total_images = len(results_foto)  # toplam görüntü sayısı
    object_counts = {}
    for obj in detected_objects:
        if obj in object_counts:
            object_counts[obj] += 1
        else:
            object_counts[obj] = 1

    # Başarı oranlarını hesaplayın ve çıktıya ekleyin
    print("\nNesne Tespit Bilgileri:")
    for obj, count in object_counts.items():
        success_rate = (count / total_images) * 100
        print(f"{obj}: {count} kez tespit edildi, Başarı Oranı: %{success_rate:.2f}")

    # Sıklıkları gösteren grafik çizimi
    plt.figure(figsize=(20, 10))
    plt.bar(object_counts.keys(), object_counts.values())
    plt.xlabel('Nesne Türü')
    plt.ylabel('Tespit Edilme Sıklığı')
    plt.title('Nesnelerin Tespit Edilme Sıklıkları')
    plt.show()

    # En sık ve nadir tespit edilen nesneleri bulmak ve yorum eklemek için
    most_frequent_object = max(object_counts, key=object_counts.get)
    least_frequent_object = min(object_counts, key=object_counts.get)
    print(f"\nEn sık tespit edilen nesne: {most_frequent_object}, Sıklık: {object_counts[most_frequent_object]}")
    print(f"En nadir tespit edilen nesne: {least_frequent_object}, Sıklık: {object_counts[least_frequent_object]}")

    # Görselleştirme ve analiz yorumu ekle
    print("Yorum: Grafik, modelin en çok ve en az hangi nesneleri tespit ettiğini göstermektedir. "
          "Model bazı nesneleri daha sık tespit ederken bazılarını daha az tespit etmiş veya hiç tespit edememiştir.")
